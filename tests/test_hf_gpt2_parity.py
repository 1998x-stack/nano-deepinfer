"""Real checkpoint correctness oracle, only when torch+transformers are installed.

The reference model forward is never used by the nano-deepinfer runner.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_deepinfer.dense_gpt2 import ReferenceGPT2  # noqa: E402
from nano_deepinfer.engine import LLMEngine  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402

MODEL = "sshleifer/tiny-gpt2"
REV = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def test_pinned_pretrained_gpt2_matches_huggingface() -> None:
    torch.set_num_threads(2)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, trust_remote_code=False, torch_dtype=torch.float32
    ).eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL, revision=REV, trust_remote_code=False
    )
    model = ReferenceGPT2.from_hf_gpt2(hf)
    prompts = ["The best way to learn systems is", "Hello, world!"]
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            ref_logits = hf(input_ids=ids, use_cache=False).logits
            actual_logits, _ = model(ids)
            err = (ref_logits - actual_logits).abs().max().item()
            assert err <= 2e-4, f"full-prefix logits differ by {err}"
            for cut in range(1, ids.shape[1]):
                _, cache = model(ids[:, :cut])
                part_logits, _ = model(ids[:, cut:], cache)
                assert torch.allclose(
                    part_logits, ref_logits[:, cut:, :], rtol=1e-3, atol=2e-4
                )
            generation = LLMEngine(
                model, model_fingerprint=REV, eos_token_id=tok.eos_token_id
            ).generate(tuple(ids[0].tolist()), SamplingParams(max_new_tokens=8))
            expected = hf.generate(
                input_ids=ids, max_new_tokens=8, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            assert generation.generated_token_ids == tuple(expected[0, ids.shape[1]:].tolist())
