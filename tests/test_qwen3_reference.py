"""Independent Qwen3 CPU oracle and HF seeded-weight agreement.

The HF implementation is used solely for testing, never for nano's forward pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_deepinfer.dense_qwen3 import (  # noqa: E402
    DenseQwen3Config,
    ReferenceQwen3,
)
from nano_deepinfer.engine import LLMEngine  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402


def _small_hf_model(*, tied: bool = False, bias: bool = False):
    torch.manual_seed(20261027)
    config = transformers.Qwen3Config(
        vocab_size=79, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, rope_theta=1000000,
        attention_bias=bias, tie_word_embeddings=tied, rms_norm_eps=1e-6,
        hidden_act="silu",
    )
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize("tied,bias", [(False, False), (True, False), (False, True)])
def test_hf_full_chunk_incremental_and_generation(tied: bool, bias: bool) -> None:
    hf = _small_hf_model(tied=tied, bias=bias)
    reference = ReferenceQwen3.from_hf_qwen3(hf)
    inputs = torch.tensor([[1, 7, 4, 12, 16, 22, 5]], dtype=torch.long)
    with torch.inference_mode():
        expected = hf(input_ids=inputs, use_cache=False).logits
        full, _ = reference(inputs)
        assert torch.allclose(expected, full, rtol=1e-3, atol=3e-4)
        for cut in (1, 3, 5):
            before, cache = reference(inputs[:, :cut])
            after, new_cache = reference(inputs[:, cut:], cache)
            assert len(new_cache) == 2
            assert new_cache[0][0].shape == (1, 2, inputs.shape[1], 8)
            assert torch.allclose(
                torch.cat((before, after), dim=1), expected, atol=3e-4, rtol=1e-3
            )
        outputs = LLMEngine(reference).generate(
            tuple(inputs[0].tolist()), SamplingParams(max_new_tokens=6, seed=42)
        )
        hf_generated = hf.generate(
            inputs, do_sample=False, max_new_tokens=6, pad_token_id=0, eos_token_id=None
        )
        assert outputs.generated_token_ids == tuple(hf_generated[0, inputs.shape[1]:].tolist())


def test_qwen_errors_and_context_limit() -> None:
    with pytest.raises(ValueError):
        DenseQwen3Config(32, 16, 32, 2, 3, 2, 8, 64)
    with pytest.raises(ValueError):
        DenseQwen3Config(32, 16, 32, 2, 4, 2, 7, 64)
    tiny = ReferenceQwen3(DenseQwen3Config(32, 16, 32, 1, 4, 2, 8, 6))
    with pytest.raises((TypeError, ValueError)):
        tiny(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        tiny(torch.ones((2, 3), dtype=torch.long))
    with pytest.raises(ValueError):
        tiny(torch.ones((1, 7), dtype=torch.long))
    _, cache = tiny(torch.ones((1, 6), dtype=torch.long))
    with pytest.raises(ValueError):
        tiny(torch.ones((1, 1), dtype=torch.long), cache)
    with pytest.raises(ValueError):
        tiny(torch.ones((1, 1), dtype=torch.long), ())
    with pytest.raises(ValueError):
        LLMEngine(tiny).generate((1, 2), SamplingParams(2), stop_token_ids=(-1,))


def test_wrong_model_or_unsupported_rope_rejected() -> None:
    hf = _small_hf_model()
    hf.config.rope_scaling = {"rope_type": "dynamic", "factor": 2.0}
    with pytest.raises(ValueError, match="RoPE"):
        ReferenceQwen3.from_hf_qwen3(hf)
