"""Tiny independent Dense GPT-2 correctness tests; CPU-only, no remote assets."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nano_deepinfer.dense_gpt2 import DenseGPT2Config, ReferenceGPT2  # noqa: E402
from nano_deepinfer.engine import LLMEngine, sample_next  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402


@pytest.fixture
def tiny_model() -> ReferenceGPT2:
    torch.manual_seed(42)
    config = DenseGPT2Config(
        vocab_size=31, hidden_size=16, num_heads=4, num_layers=2, max_positions=20
    )
    return ReferenceGPT2(config).eval()


def test_full_forward_vs_incremental_cache(tiny_model: ReferenceGPT2) -> None:
    prompt = torch.tensor([[1, 3, 5, 7, 9, 11]], dtype=torch.long)
    with torch.inference_mode():
        full, _ = tiny_model(prompt)
        prefix, cache = tiny_model(prompt[:, :2])
        logits = [(1, prefix[0, -1])]
        for i in range(2, prompt.shape[1]):
            step, cache = tiny_model(prompt[:, i : i + 1], cache)
            logits.append((i, step[0, -1]))
        assert len(cache) == 2
        assert cache[0][0].shape == (1, 4, 6, 4)
        for position, step in logits:
            assert torch.allclose(step, full[0, position], atol=2e-5, rtol=1e-5)


def test_full_forward_vs_several_prefill_segments(tiny_model: ReferenceGPT2) -> None:
    x = torch.tensor([[4, 6, 8, 10, 12]], dtype=torch.long)
    with torch.inference_mode():
        full, _ = tiny_model(x)
        part_a, a = tiny_model(x[:, :3])
        part_b, b = tiny_model(x[:, 3:5], a)
        assert torch.allclose(full, torch.cat([part_a, part_b], dim=1), atol=2e-5)
        assert b[0][0].shape[-2] == 5


def test_generate_matches_direct_greedy(tiny_model: ReferenceGPT2) -> None:
    prompt = (1, 3, 5)
    result = LLMEngine(tiny_model).generate(prompt, SamplingParams(max_new_tokens=6))
    tokens = list(prompt)
    for token in result.generated_token_ids:
        logits, _ = tiny_model(torch.tensor([tokens], dtype=torch.long))
        assert token == int(logits[0, -1].argmax().item())
        tokens.append(token)
    assert len(result.generated_token_ids) == 6
    assert result.finish_reason == "length"


def test_seeded_sampling_and_valid_probabilities(tiny_model: ReferenceGPT2) -> None:
    engine = LLMEngine(tiny_model)
    params = SamplingParams(max_new_tokens=6, temperature=0.9, top_k=5, top_p=0.8, seed=10)
    a = engine.generate((2, 4, 6), params)
    b = engine.generate((2, 4, 6), params)
    assert a.generated_token_ids == b.generated_token_ids
    assert len(a.generated_token_ids) == 6


@pytest.mark.parametrize("seed", [0, 1, 10, 999])
def test_top_k_single_candidate_is_argmax(seed: int) -> None:
    logits = torch.tensor([1.0, -4.0, 2.0, 0.0])
    token = sample_next(logits, SamplingParams(1, temperature=0.7, top_k=1, seed=seed),
                        torch.Generator().manual_seed(seed))
    assert token == 2


def test_eos_stops_correctly(tiny_model: ReferenceGPT2) -> None:
    initial, _ = tiny_model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    eos = int(initial[0, -1].argmax().item())
    result = LLMEngine(tiny_model, eos_token_id=eos).generate(
        (1, 2, 3), SamplingParams(max_new_tokens=6)
    )
    assert result.generated_token_ids == (eos,)
    assert result.finish_reason == "eos"


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([], dtype=torch.long).reshape(1, 0),
        torch.tensor([[31]], dtype=torch.long),
        torch.tensor([[-1]], dtype=torch.long),
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.ones((2, 3), dtype=torch.long),
    ],
)
def test_invalid_input_rejected(tiny_model: ReferenceGPT2, tokens: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        tiny_model(tokens)


def test_context_limit_and_cache_consistency(tiny_model: ReferenceGPT2) -> None:
    input_ids = torch.ones((1, 20), dtype=torch.long)
    _, kv = tiny_model(input_ids)
    with pytest.raises(ValueError):
        tiny_model(torch.ones((1, 1), dtype=torch.long), kv)
    with pytest.raises(ValueError):
        tiny_model(torch.ones((1, 1), dtype=torch.long), kv[:1])


def test_config_checks() -> None:
    with pytest.raises(ValueError):
        DenseGPT2Config(vocab_size=10, hidden_size=15, num_heads=4, num_layers=1,
                        max_positions=10)
    with pytest.raises(ValueError):
        DenseGPT2Config(vocab_size=0, hidden_size=16, num_heads=4, num_layers=1,
                        max_positions=10)
