"""Stage 02 deterministic model-based, fault, threading and leak tests."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from nano_deepinfer.cache.block_manager import (
    BlockHandle,
    BlockManager,
    KVCapacityError,
    KVStateError,
)


def test_cow_abort_commit_generation_and_idempotent_release() -> None:
    manager = BlockManager(5, 4)
    manager.create("a")
    first = manager.begin("a", 3)
    manager.commit(first)
    old = manager.table("a").blocks[0]
    assert manager.is_live(old)
    tx = manager.begin("a", 4)
    assert tx.proposed[0] != old
    assert manager.table("a").num_tokens == 3
    manager.abort(tx)
    assert manager.table("a").blocks == (old,)
    with pytest.raises(KVStateError):
        manager.commit(tx)
    tx = manager.begin("a", 4)
    next_page = tx.proposed[0]
    manager.commit(tx)
    assert not manager.is_live(old)
    assert manager.is_live(next_page)
    with pytest.raises(KVStateError):
        manager.commit(tx)
    manager.assert_consistent()
    assert manager.release("a")
    assert not manager.release("a")
    assert not manager.is_live(next_page)
    assert manager.stats().free == 5
    manager.create("b")
    manager.commit(manager.begin("b", 4))
    # Generation prevents an old handle from observing re-used physical bytes.
    assert not manager.is_live(old)
    assert not manager.is_live(next_page)
    manager.assert_consistent()


def test_atomic_oom_does_not_mutate_committed_state() -> None:
    mgr = BlockManager(1, 4)
    mgr.create("r")
    mgr.commit(mgr.begin("r", 3))
    before = mgr.table("r")
    with pytest.raises(KVCapacityError):
        mgr.begin("r", 4)  # COW needs a spare page for the last partial block.
    assert mgr.table("r") == before
    assert mgr.stats().pending == 0
    mgr.assert_consistent()
    mgr.release("r")
    assert mgr.stats().used == 0


@pytest.mark.parametrize("blocks,size", [(1, 1), (4, 2), (11, 4), (17, 16)])
def test_randomized_allocator_against_independent_length_oracle(blocks: int, size: int) -> None:
    for seed in range(12):
        rng = random.Random(seed * 1000 + blocks)
        mgr = BlockManager(blocks, size)
        lengths: dict[str, int] = {}
        for _ in range(180):
            choice = rng.randrange(4)
            ident = f"r{rng.randrange(9)}"
            if choice == 0 and ident not in lengths:
                mgr.create(ident)
                lengths[ident] = 0
            elif choice == 1 and ident in lengths:
                target = lengths[ident] + rng.randrange(size * 2 + 1)
                try:
                    tx = mgr.begin(ident, target)
                except KVCapacityError:
                    assert mgr.table(ident).num_tokens == lengths[ident]
                else:
                    assert mgr.table(ident).num_tokens == lengths[ident]
                    if rng.randrange(4) == 0:
                        mgr.abort(tx)
                    else:
                        mgr.commit(tx)
                        lengths[ident] = target
            elif choice == 2 and ident in lengths:
                mgr.release(ident)
                del lengths[ident]
            # Keep a separate live-token oracle: memory manager is the SUT.
            assert mgr.stats().owners == len(lengths)
            for rid, length in lengths.items():
                table = mgr.table(rid)
                assert table.num_tokens == length
                assert len(table.blocks) == (length + size - 1) // size
            mgr.assert_consistent()
        for rid in list(lengths):
            mgr.release(rid)
        mgr.assert_consistent()
        assert mgr.stats().used == 0
        assert mgr.stats().free == blocks


def test_pending_cancel_reclaims_both_old_and_new_pages() -> None:
    manager = BlockManager(8, 4)
    manager.create("request")
    manager.commit(manager.begin("request", 3))
    tx = manager.begin("request", 7)
    pages = tx.proposed
    assert manager.stats().pending == 1
    assert manager.release("request")
    assert manager.stats().free == 8
    assert all(not manager.is_live(page) for page in pages)
    with pytest.raises(KVStateError):
        manager.abort(tx)
    manager.assert_consistent()


def test_invalid_inputs_and_stale_reservations() -> None:
    with pytest.raises(ValueError):
        BlockManager(0, 4)
    with pytest.raises(ValueError):
        BlockManager(1, 0)
    manager = BlockManager(3, 4)
    with pytest.raises(ValueError):
        manager.create("")
    manager.create("r")
    with pytest.raises(KVStateError):
        manager.create("r")
    with pytest.raises(ValueError):
        manager.begin("r", -1)
    tx = manager.begin("r", 6)
    with pytest.raises(KVStateError):
        manager.begin("r", 7)
    with pytest.raises(KVStateError):
        manager.commit(
            type(tx)(tx.request_id, tx.reservation_id, tx.target_tokens,
                     tx.original, tx.proposed, tx.fresh)
        )
    manager.abort(tx)
    with pytest.raises(KVStateError):
        manager.commit(tx)
    with pytest.raises(KVStateError):
        manager.table("unknown")
    assert not manager.is_live(BlockHandle(-1, 1))
    manager.assert_consistent()


def test_threaded_unique_request_allocation_and_cleanup() -> None:
    manager = BlockManager(96, 4)

    def worker(idx: int) -> None:
        rid = f"worker-{idx}"
        manager.create(rid)
        for length in (3, 4, 7, 8, 10, 12):
            tx = manager.begin(rid, length)
            manager.commit(tx)
        assert manager.release(rid)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(8)))
    manager.assert_consistent()
    assert manager.stats().free == 96
