"""Stage 02 CPU ownership oracle, randomized state machine and failure tests."""

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from nano_deepinfer.cache.block_manager import (
    BlockHandle,
    BlockManager,
    KVCapacityError,
    KVStateError,
)


def test_atomic_cow_reserve_abort_commit_and_generations() -> None:
    m = BlockManager(4, 4)
    m.create("a")
    initial = m.begin("a", 3)
    m.commit(initial)
    old = m.table("a").blocks[0]
    tx = m.begin("a", 4)
    assert tx.proposed != (old,)
    assert m.table("a").blocks == (old,)
    m.abort(tx)
    with pytest.raises(KVStateError):
        m.commit(tx)
    assert m.is_live(old)
    tx = m.begin("a", 4)
    m.commit(tx)
    assert not m.is_live(old)
    assert m.release("a") and not m.release("a")
    with pytest.raises(KVStateError):
        m.commit(tx)
    m.create("b")
    m.commit(m.begin("b", 4))
    assert not m.is_live(old)
    assert not m.is_live(BlockHandle(-1, 1))
    m.assert_consistent()


def test_oom_does_not_modify_committed_state() -> None:
    m = BlockManager(1, 4)
    m.create("request")
    m.commit(m.begin("request", 3))
    before = m.table("request")
    with pytest.raises(KVCapacityError):
        m.begin("request", 4)  # conservative COW requires an extra page
    assert m.table("request") == before
    assert m.stats().pending == 0
    m.assert_consistent()


@pytest.mark.parametrize("blocks,size", [(1, 1), (4, 2), (9, 4), (17, 8)])
def test_seeded_randomized_model_based_allocator(blocks: int, size: int) -> None:
    for seed in range(12):
        rng = random.Random(seed * 100 + blocks)
        m = BlockManager(blocks, size)
        oracle: dict[str, int] = {}
        for _ in range(200):
            rid = f"r{rng.randrange(8)}"
            decision = rng.randrange(4)
            if decision == 0 and rid not in oracle:
                m.create(rid)
                oracle[rid] = 0
            elif decision == 1 and rid in oracle:
                target = oracle[rid] + rng.randrange(size * 2 + 1)
                try:
                    tx = m.begin(rid, target)
                except KVCapacityError:
                    assert m.table(rid).num_tokens == oracle[rid]
                else:
                    if rng.randrange(4) == 0:
                        m.abort(tx)
                    else:
                        m.commit(tx)
                        oracle[rid] = target
            elif decision == 2 and rid in oracle:
                assert m.release(rid)
                del oracle[rid]
            assert m.stats().owners == len(oracle)
            for name, num_tokens in oracle.items():
                table = m.table(name)
                assert table.num_tokens == num_tokens
                assert len(table.blocks) == (num_tokens + size - 1) // size
            m.assert_consistent()
        for rid in list(oracle):
            m.release(rid)
        assert m.stats().used == 0
        m.assert_consistent()


def test_cancel_pending_reclaims_all_blocks() -> None:
    m = BlockManager(8, 4)
    m.create("x")
    m.commit(m.begin("x", 3))
    pending = m.begin("x", 8)
    assert m.release("x")
    assert m.stats().free == 8
    assert not any(m.is_live(h) for h in pending.proposed)
    with pytest.raises(KVStateError):
        m.abort(pending)
    m.assert_consistent()


def test_invalid_requests_and_stale_transaction_rejected() -> None:
    with pytest.raises(ValueError):
        BlockManager(0, 2)
    m = BlockManager(4, 2)
    with pytest.raises(ValueError):
        m.create("")
    m.create("one")
    with pytest.raises(KVStateError):
        m.create("one")
    with pytest.raises(ValueError):
        m.begin("one", -1)
    tx = m.begin("one", 3)
    with pytest.raises(KVStateError):
        m.begin("one", 4)
    with pytest.raises(KVStateError):
        m.commit(type(tx)(
            tx.request_id, tx.reservation_id, tx.target_tokens,
            tx.original, tx.proposed, tx.fresh
        ))
    m.abort(tx)
    with pytest.raises(KVStateError):
        m.commit(tx)
    with pytest.raises(KVStateError):
        m.table("does-not-exist")
    m.assert_consistent()


def test_thread_safe_create_grow_and_release() -> None:
    m = BlockManager(96, 4)

    def work(index: int) -> None:
        rid = f"worker-{index}"
        m.create(rid)
        for total in (3, 4, 7, 8, 10, 12):
            m.commit(m.begin(rid, total))
        assert m.release(rid)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(work, range(8)))
    assert m.stats().free == 96
    m.assert_consistent()
