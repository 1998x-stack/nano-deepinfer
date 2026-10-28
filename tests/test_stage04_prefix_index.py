"""Stage 04 pure-Python physical refcounts, namespace isolation and eviction."""

from dataclasses import replace

import pytest

from nano_deepinfer.cache.block_manager import (
    BlockManager,
    KVStateError,
)
from nano_deepinfer.cache.prefix_cache import CacheNamespace


def scope(**overrides: str) -> CacheNamespace:
    return CacheNamespace(
        model_revision=overrides.get("model_revision", "immutable-sha"),
        tokenizer_revision=overrides.get("tokenizer_revision", "tokenizer-v1"),
        positional_config=overrides.get("positional_config", "rope-v1"),
        tenant_id=overrides.get("tenant_id", "org-a"),
        dtype=overrides.get("dtype", "torch.float32"),
        adapter_revision=overrides.get("adapter_revision", "none"),
    )


def test_namespace_digest_is_content_and_isolation_sensitive() -> None:
    base = scope()
    tokens = (1, 2, 3, 4)
    key = base.key_for(tokens)
    assert len(key) == 64
    assert key == base.key_for(tokens)
    assert key != base.key_for((1, 2, 3, 5))
    for field in ("model_revision", "tokenizer_revision", "positional_config",
                  "tenant_id", "dtype", "adapter_revision", "cache_schema"):
        assert key != replace(base, **{field: "other"}).key_for(tokens)
    assert "org-a" not in key
    with pytest.raises(ValueError):
        scope(tenant_id="")
    with pytest.raises(ValueError):
        base.key_for((-1, 2))


def test_shared_block_pin_attach_tail_cow_and_release() -> None:
    m = BlockManager(5, 4)
    m.create("producer")
    m.commit(m.begin("producer", 4))
    saved = m.table("producer").blocks
    assert m.ref_count(saved[0]) == 1
    m.pin(saved)
    assert m.ref_count(saved[0]) == 2
    attached = m.attach_prefix("consumer", saved, 4)
    assert attached.blocks == saved
    assert m.ref_count(saved[0]) == 3
    m.release("producer")
    assert m.ref_count(saved[0]) == 2
    # Append to a shared full prefix: new block only, no mutation of shared one.
    res = m.begin("consumer", 6)
    m.commit(res)
    assert m.table("consumer").blocks[0] == saved[0]
    assert m.ref_count(saved[0]) == 2
    m.unpin(saved)
    assert m.ref_count(saved[0]) == 1
    assert m.release("consumer")
    assert not m.is_live(saved[0])
    assert m.stats().used == 0
    m.assert_consistent()


def test_shared_partial_tail_is_never_published_as_a_prefix() -> None:
    m = BlockManager(5, 4)
    m.create("owner")
    m.commit(m.begin("owner", 3))
    block = m.table("owner").blocks
    with pytest.raises(KVStateError):
        m.attach_prefix("invalid", block, 3)
    assert m.ref_count(block[0]) == 1
    m.assert_consistent()
    m.release("owner")


def test_stale_prefix_and_unbalanced_pin_are_rejected() -> None:
    m = BlockManager(3, 2)
    m.create("a")
    m.commit(m.begin("a", 2))
    old = m.table("a").blocks
    m.pin(old)
    m.unpin(old)
    with pytest.raises(KVStateError):
        m.unpin(old)
    m.release("a")
    assert not m.is_live(old[0])
    with pytest.raises(KVStateError):
        m.pin(old)
    with pytest.raises(KVStateError):
        m.attach_prefix("new", old, 2)
    m.assert_consistent()


def test_pin_accounting_makes_used_count_physical_not_total_refs() -> None:
    m = BlockManager(2, 2)
    m.create("a")
    m.commit(m.begin("a", 2))
    block = m.table("a").blocks
    m.pin(block)
    m.attach_prefix("b", block, 2)
    assert m.stats().used == 1
    assert m.stats().free == 1
    m.assert_consistent()
    m.release("a")
    m.release("b")
    assert m.stats().used == 1
    m.unpin(block)
    assert m.stats().free == 2
    m.assert_consistent()

