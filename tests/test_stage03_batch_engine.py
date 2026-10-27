"""Stage 03 multi-request CPU model oracle, eviction and fault tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nano_deepinfer.batch_engine import ContinuousBatchEngine  # noqa: E402
from nano_deepinfer.cache.paged_kv import PagedKVStore  # noqa: E402
from nano_deepinfer.dense_gpt2 import DenseGPT2Config, ReferenceGPT2  # noqa: E402
from nano_deepinfer.dense_qwen3 import DenseQwen3Config, ReferenceQwen3  # noqa: E402
from nano_deepinfer.engine import LLMEngine  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402


def make_engine(
    family: str = "gpt2", pages: int = 12, budget: int = 7, slots: int = 3,
) -> ContinuousBatchEngine:
    torch.manual_seed(311)
    if family == "gpt2":
        model = ReferenceGPT2(DenseGPT2Config(43, 16, 4, 1, 40)).eval()
        cache = PagedKVStore(pages, 4, 1, 4, 4)
    else:
        model = ReferenceQwen3(DenseQwen3Config(
            vocab_size=43, hidden_size=32, intermediate_size=64,
            num_layers=1, num_heads=4, num_kv_heads=2,
            head_dim=8, max_positions=40,
        )).eval()
        cache = PagedKVStore(pages, 4, 1, 2, 8)
    return ContinuousBatchEngine(
        model, cache, max_num_scheduled_tokens=budget, max_num_seqs=slots
    )


@pytest.mark.parametrize("family", ["gpt2", "qwen3"])
@pytest.mark.parametrize("temp", [0.0, 0.65])
def test_multi_request_results_equal_isolated_reference(family: str, temp: float) -> None:
    engine = make_engine(family=family, pages=32, budget=7, slots=3)
    params = SamplingParams(max_new_tokens=4, temperature=temp, top_k=9, seed=101)
    prompts = {
        "a": (1, 2, 3), "b": (3, 4, 5, 6), "c": (8, 9, 3),
        "d": (5, 2), "e": (2, 3, 4, 5, 6),
    }
    isolated = {
        rid: LLMEngine(engine.model).generate(prompt, params).generated_token_ids
        for rid, prompt in prompts.items()
    }
    for rid, prompt in prompts.items():
        engine.add_request(rid, prompt, params)
    result = engine.run(max_steps=100)
    assert set(result) == set(prompts)
    for rid in prompts:
        assert result[rid].generated_token_ids == isolated[rid]
        assert result[rid].finish_reason == "length"
    assert engine.cache.manager.stats().used == 0
    engine.assert_consistent()
    assert any(evt.kind == "decode" for step in engine.history for evt in step.events)
    assert any(evt.kind == "prefill" for step in engine.history for evt in step.events)
    assert all(step.scheduled_tokens <= step.token_budget for step in engine.history)


def test_admission_during_decode_and_cancellation_no_leak() -> None:
    engine = make_engine(pages=8, budget=3, slots=2)
    params = SamplingParams(max_new_tokens=4)
    engine.add_request("old", (1, 2, 3), params)
    engine.step()  # prefill old
    engine.step()  # decode old
    engine.add_request("new", (2, 3, 4), params)
    engine.add_request("cancel", (1, 2, 3), params)
    assert engine.cancel("cancel")
    assert not engine.cancel("cancel")
    result = engine.run()
    assert result["cancel"].finish_reason == "cancelled"
    assert len(result["new"].generated_token_ids) == 4
    assert len(result["old"].generated_token_ids) == 4
    assert engine.cache.manager.stats().used == 0
    engine.assert_consistent()


def test_eviction_recompute_is_token_exact_and_not_lost() -> None:
    engine = make_engine(pages=2, budget=5, slots=2)
    params = SamplingParams(max_new_tokens=3, temperature=0.0, seed=1)
    prompts = {"a": (1, 2), "b": (3, 4)}
    expected = {
        rid: LLMEngine(engine.model).generate(p, params).generated_token_ids
        for rid, p in prompts.items()
    }
    for rid, p in prompts.items():
        engine.add_request(rid, p, params)
    result = engine.run(max_steps=100)
    assert {rid: val.generated_token_ids for rid, val in result.items()} == expected
    assert engine.preemption_count > 0
    assert engine.cache.manager.stats().used == 0
    assert any(evt.kind == "preempt" for step in engine.history for evt in step.events)
    engine.assert_consistent()


def test_no_possible_page_capacity_marks_request_failed_and_frees_pages() -> None:
    engine = make_engine(pages=1, budget=6, slots=1)
    engine.add_request("oom", (1, 2, 3), SamplingParams(max_new_tokens=3))
    result = engine.run()
    assert result["oom"].finish_reason == "failed"
    assert engine.cache.manager.stats().used == 0
    engine.assert_consistent()


def test_bad_input_does_not_mutate_queue_and_duplicate_ids_fail() -> None:
    engine = make_engine()
    params = SamplingParams(max_new_tokens=1)
    with pytest.raises(ValueError):
        engine.add_request("bad", (1,) * 44, params)
    with pytest.raises(ValueError):
        engine.add_request("bad", (1,), params, stop_token_ids=(-1,))
    engine.add_request("good", (1,), params)
    with pytest.raises(ValueError):
        engine.add_request("good", (1,), params)
    assert engine.cancel("good")
    with pytest.raises(ValueError):
        engine.add_request("good", (1,), params)
    assert engine.pending == 0
    engine.assert_consistent()


def test_no_resource_leak_on_model_failure_and_other_requests_continue() -> None:
    engine = make_engine()
    original_forward = engine.model.forward

    def injected_forward(tokens, cache=None):
        if int(tokens[0, 0].item()) == 7:
            raise RuntimeError("injected model crash")
        return original_forward(tokens, cache)

    engine.model.forward = injected_forward
    params = SamplingParams(max_new_tokens=2)
    engine.add_request("failure", (7, 2), params)
    engine.add_request("healthy", (2, 3), params)
    result = engine.run()
    assert result["failure"].finish_reason == "failed"
    assert len(result["healthy"].generated_token_ids) == 2
    assert engine.cache.manager.stats().free == engine.cache.manager.num_blocks
    engine.assert_consistent()


def test_100_requests_complete_without_deadlock_or_duplicate_token() -> None:
    engine = make_engine(pages=10, budget=8, slots=3)
    params = SamplingParams(max_new_tokens=2, seed=42)
    for i in range(100):
        engine.add_request(f"job-{i:03d}", (1 + i % 8, 2), params)
    results = engine.run(max_steps=2000)
    assert len(results) == 100
    assert all(len(result.generated_token_ids) == 2 for result in results.values())
    assert all(result.finish_reason == "length" for result in results.values())
    assert engine.cache.manager.stats().free == 10
    engine.assert_consistent()


def test_bounded_admission_backpressure() -> None:
    engine = make_engine()
    engine.max_pending_requests = 2
    params = SamplingParams(max_new_tokens=1)
    engine.add_request("a", (1, 2), params)
    engine.add_request("b", (1, 2), params)
    with pytest.raises(RuntimeError, match="backpressure"):
        engine.add_request("c", (1, 2), params)
    assert engine.cancel("a")
    engine.add_request("c", (1, 2), params)
    assert engine.pending == 2
    engine.run()
    engine.assert_consistent()
