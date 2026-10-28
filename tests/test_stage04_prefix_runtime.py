"""Stage 04 physically shared KV blocks, correctness, LRU and request isolation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nano_deepinfer.batch_engine import ContinuousBatchEngine  # noqa: E402
from nano_deepinfer.cache.paged_kv import PagedKVStore  # noqa: E402
from nano_deepinfer.cache.prefix_cache import CacheNamespace  # noqa: E402
from nano_deepinfer.dense_gpt2 import DenseGPT2Config, ReferenceGPT2  # noqa: E402
from nano_deepinfer.dense_qwen3 import DenseQwen3Config, ReferenceQwen3  # noqa: E402
from nano_deepinfer.engine import LLMEngine  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402


def setup(family: str = "gpt2", blocks: int = 16, cached: int = 3,
          chunk: int = 4) -> tuple[ContinuousBatchEngine, CacheNamespace]:
    torch.manual_seed(2026)
    if family == "gpt2":
        model = ReferenceGPT2(DenseGPT2Config(57, 16, 4, 2, 32)).eval()
        store = PagedKVStore(blocks, 4, 2, 4, 4, max_cached_blocks=cached)
    else:
        model = ReferenceQwen3(DenseQwen3Config(57, 32, 64, 2, 4, 2, 8, 32)).eval()
        store = PagedKVStore(blocks, 4, 2, 2, 8, max_cached_blocks=cached)
    scope = CacheNamespace("model-2026", "tok-1", "arch-1", "tenant-A", str(store.dtype))
    runner = ContinuousBatchEngine(
        model, store, model_fingerprint="model-2026",
        max_num_scheduled_tokens=8, max_num_seqs=2,
        max_prefill_chunk_size=chunk, prefix_namespace=scope,
    )
    return runner, scope


@pytest.mark.parametrize("family", ["gpt2", "qwen3"])
@pytest.mark.parametrize("sampling", [0.0, 0.7])
def test_shared_prefix_and_suffix_exact_match_uncached(family: str, sampling: float) -> None:
    engine, scope = setup(family)
    params = SamplingParams(max_new_tokens=4, temperature=sampling, seed=14, top_k=8)
    prompt_a = (2, 4, 6, 8)
    prompt_b = (2, 4, 6, 8, 10, 12)
    first = LLMEngine(engine.model).generate(prompt_a, params).generated_token_ids
    second = LLMEngine(engine.model).generate(prompt_b, params).generated_token_ids
    engine.add_request("a", prompt_a, params)
    engine.run()
    assert engine.results["a"].generated_token_ids == first
    assert engine.cache.prefix_cache.entries >= 1
    assert engine.cache.manager.stats().used >= 1  # Cache pins survive request A.
    engine.add_request("b", prompt_b, params)
    assert engine.prefix_hits >= 1
    assert engine.reused_prefix_tokens >= 4
    table = engine.cache.manager.table("b")
    assert table.num_tokens == 4
    assert engine.cache.manager.ref_count(table.blocks[0]) >= 2
    engine.run()
    assert engine.results["b"].generated_token_ids == second
    assert engine.cache.prefix_cache.hits >= 1
    engine.cache.assert_consistent()
    engine.cache.clear_prefix()
    assert engine.cache.manager.stats().used == 0


def test_exact_prompt_match_reuses_last_logits_no_forward_for_prefill() -> None:
    engine, _ = setup()
    params = SamplingParams(max_new_tokens=2, seed=0)
    prompt = (2, 4, 6, 8)
    engine.add_request("a", prompt, params)
    engine.run()
    events_before = len(engine.history)
    engine.add_request("b", prompt, params)
    assert engine._states["b"].computed == len(prompt)
    assert engine._states["b"].last_logits is not None
    engine.run()
    assert engine.results["a"].generated_token_ids == engine.results["b"].generated_token_ids
    assert not any(
        e.request_id == "b" and e.kind == "prefill"
        for step in engine.history[events_before:] for e in step.events
    )
    engine.cache.clear_prefix()
    engine.cache.assert_consistent()


def test_cross_tenant_cannot_attach_existing_prefix() -> None:
    engine, scope = setup()
    engine.add_request("a", (2, 4, 6, 8), SamplingParams(max_new_tokens=1))
    engine.run()
    other = CacheNamespace(
        scope.model_revision, scope.tokenizer_revision, scope.positional_config,
        "tenant-B", scope.dtype, scope.adapter_revision,
    )
    assert engine.cache.attach_prefix("b", other, (2, 4, 6, 8)) is None
    assert engine.cache.manager.stats().owners == 0
    assert engine.cache.clear_prefix(tenant_id="tenant-B") == 0
    assert engine.cache.clear_prefix(tenant_id="tenant-A") == 1
    engine.cache.assert_consistent()


def test_lru_eviction_and_inflight_reader_remains_live() -> None:
    engine, scope = setup(blocks=5, cached=1)
    params = SamplingParams(max_new_tokens=1)
    engine.add_request("a", (1, 2, 3, 4), params)
    engine.run()
    assert engine.cache.prefix_cache.entries == 1
    saved = engine.cache.prefix_cache.cached_blocks
    assert saved == 1
    # Attach an in-flight reader before cache index is evicted.
    hit = engine.cache.attach_prefix("reader", scope, (1, 2, 3, 4))
    assert hit is not None
    reader_table = engine.cache.manager.table("reader")
    old = reader_table.blocks[0]
    engine.add_request("b", (3, 4, 5, 6), params)
    engine.run()
    assert engine.cache.prefix_cache.evictions >= 1
    # Reader-owned page cannot be freed just by LRU eviction.
    assert engine.cache.manager.is_live(old)
    assert engine.cache.read("reader")[0][0].shape[2] == 4
    engine.cache.release("reader")
    engine.cache.clear_prefix()
    assert engine.cache.manager.stats().used == 0


def test_prefill_chunk_cap_and_prompt_replay() -> None:
    engine, _ = setup(chunk=2)
    params = SamplingParams(max_new_tokens=1)
    engine.add_request("a", (1, 2, 3, 4), params)
    engine.run()
    prefill = [
        e.cost for s in engine.history for e in s.events
        if e.request_id == "a" and e.kind == "prefill"
    ]
    assert prefill == [2, 2]
    engine.cache.clear_prefix()


def test_disabled_prefix_feature_retains_old_behavior() -> None:
    torch.manual_seed(123)
    model = ReferenceGPT2(DenseGPT2Config(57, 16, 4, 2, 32)).eval()
    cache = PagedKVStore(12, 4, 2, 4, 4)
    engine = ContinuousBatchEngine(model, cache)
    assert cache.prefix_cache is None
    engine.add_request("a", (1, 2, 3, 4), SamplingParams(max_new_tokens=2))
    engine.run()
    assert cache.manager.stats().used == 0
    assert engine.prefix_hits == 0


def test_namespace_mismatch_rejected_on_engine_init() -> None:
    torch.manual_seed(2)
    model = ReferenceGPT2(DenseGPT2Config(57, 16, 4, 2, 32)).eval()
    cache = PagedKVStore(12, 4, 2, 4, 4, max_cached_blocks=2)
    wrong = CacheNamespace("wrong-weights", "tok", "pos", "tenant", str(cache.dtype))
    with pytest.raises(ValueError, match="namespace"):
        ContinuousBatchEngine(model, cache, model_fingerprint="actual", prefix_namespace=wrong)
    with pytest.raises(ValueError, match="namespace"):
        ContinuousBatchEngine(model, cache, model_fingerprint="actual")

