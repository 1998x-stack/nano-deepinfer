"""Pure Python Stage 03 scheduling tests: no Torch/HF model dependency."""

import pytest

from nano_deepinfer.scheduler import RequestView, UnifiedScheduler


def view(rid: str, computed: int = 0, total: int = 4, version: int = 0) -> RequestView:
    return RequestView(rid, total, computed, version)


def test_mixed_prefill_and_decode_share_strict_token_budget() -> None:
    sched = UnifiedScheduler(max_num_scheduled_tokens=5, max_num_seqs=3)
    for name in ("a", "b", "c"):
        sched.add(name)
    batch = sched.plan({"a": view("a", 0, 3), "b": view("b", 3, 3), "c": view("c", 0, 4)})
    assert batch.used_tokens == 5
    assert [item.kind for item in batch.work] == ["prefill", "decode", "prefill"]
    assert [item.token_count for item in batch.work] == [3, 1, 1]
    assert [item.input_offset for item in batch.work] == [0, 3, 0]


def test_round_robin_prevents_starvation_when_only_one_slot() -> None:
    sched = UnifiedScheduler(8, 1)
    names = tuple(f"r{i}" for i in range(11))
    for rid in names:
        sched.add(rid)
    views = {rid: view(rid, 4, 4) for rid in names}
    cycles = [sched.plan(views).work[0].request_id for _ in range(33)]
    assert cycles[:11] == list(names)
    assert cycles[11:22] == list(names)
    assert cycles[22:] == list(names)


def test_100_request_queue_can_make_progress_with_tiny_budget() -> None:
    sched = UnifiedScheduler(1, 2)
    for i in range(100):
        sched.add(f"q{i}")
    active = {f"q{i}": view(f"q{i}", 0, 9) for i in range(100)}
    first = [sched.plan(active).work[0].request_id for _ in range(100)]
    assert set(first) == set(active)
    assert len(first) == 100
    for _ in range(100):
        batch = sched.plan(active)
        assert batch.used_tokens == 1
        assert len(batch.work) == 1
        assert batch.work[0].token_count == 1


def test_scheduling_chunk_offset_and_recompute_after_preempt() -> None:
    sched = UnifiedScheduler(3, 1)
    sched.add("a")
    first = sched.plan({"a": view("a", 0, 10)})
    assert first.work[0].input_offset == 0
    assert first.work[0].token_count == 3
    next_step = sched.plan({"a": view("a", 3, 10, 1)})
    assert next_step.work[0].input_offset == 3
    replay = sched.plan({"a": view("a", 0, 10, 2)})
    assert replay.work[0].input_offset == 0
    assert replay.work[0].version == 2


def test_duplicate_unknown_and_invalid_states() -> None:
    sched = UnifiedScheduler(2, 1)
    sched.add("r")
    with pytest.raises(ValueError):
        sched.add("r")
    with pytest.raises(ValueError):
        sched.plan({"unknown": view("unknown")})
    with pytest.raises(ValueError):
        view("r", 10, 2)
    with pytest.raises(ValueError):
        UnifiedScheduler(0, 1)
    assert sched.remove("r")
    assert not sched.remove("r")
    assert sched.pending == 0
    assert sched.plan({}).work == ()


def test_decode_can_run_while_new_request_prefills() -> None:
    sched = UnifiedScheduler(8, 3)
    sched.add("existing")
    assert sched.plan({"existing": view("existing", 3, 3)}).work[0].kind == "decode"
    sched.add("new")
    batch = sched.plan({"existing": view("existing", 3, 3), "new": view("new", 0, 4)})
    assert {work.kind for work in batch.work} == {"decode", "prefill"}
    assert batch.used_tokens == 5
