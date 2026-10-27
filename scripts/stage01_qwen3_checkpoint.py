# mypy: ignore-errors
"""Stage 01 real tiny Qwen3 checkpoint oracle; only tested revision, not performance.

The published checkpoint is random debugging weights, not a useful text generator.
This script pins/records the resolved full Hugging Face commit SHA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_deepinfer.foundation import SamplingParams, environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch  # type: ignore[import-not-found]
    from huggingface_hub import HfApi  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    from nano_deepinfer.dense_qwen3 import ReferenceQwen3
    from nano_deepinfer.engine import LLMEngine

    torch.manual_seed(0)
    torch.set_num_threads(2)
    repo = "llamafactory/tiny-random-qwen3"
    # A known immutable revision prefix (verified and expanded to full SHA by Hub).
    revision_prefix = "81d6f5f"
    full_revision = HfApi().model_info(repo, revision=revision_prefix).sha
    if full_revision is None or len(full_revision) != 40 or not full_revision.startswith(revision_prefix):
        raise AssertionError("Model revision was not resolved to the intended immutable commit")
    hf = AutoModelForCausalLM.from_pretrained(
        repo, revision=full_revision, torch_dtype=torch.float32, trust_remote_code=False
    ).eval()
    nano = ReferenceQwen3.from_hf_qwen3(hf)
    cases = []
    for prompt in ((1, 15, 32, 73), (4, 2, 89, 27, 11, 35)):
        ids = torch.tensor([prompt], dtype=torch.long)
        with torch.inference_mode():
            expected = hf(ids, use_cache=False).logits
            actual, cache = nano(ids)
            delta = (expected - actual).abs().max().item()
            if delta >= 2e-3:
                raise AssertionError(f"Qwen3 full-logits mismatch: {delta}")
            prefill, kv = nano(ids[:, :2])
            decode, _ = nano(ids[:, 2:], kv)
            cache_delta = (torch.cat((prefill, decode), dim=1) - expected).abs().max().item()
            if cache_delta >= 2e-3:
                raise AssertionError(f"Qwen3 cached-logits mismatch: {cache_delta}")
            result = LLMEngine(
                nano, model_fingerprint=full_revision, eos_token_id=None
            ).generate(prompt, SamplingParams(max_new_tokens=5))
            reference_ids = hf.generate(
                ids, do_sample=False, max_new_tokens=5, pad_token_id=0, eos_token_id=None
            )[0, len(prompt):].tolist()
            if list(result.generated_token_ids) != reference_ids:
                raise AssertionError("Generated IDs disagree with Hugging Face")
        cases.append({
            "prompt_token_ids": list(prompt), "reference_ids": reference_ids,
            "nano_ids": list(result.generated_token_ids),
            "max_abs_logits_error": delta, "max_abs_cached_logits_error": cache_delta,
            "kv_shape": list(cache[0][0].shape),
        })
    report = {
        "schema_version": 1, "stage": "01", "status": "passed",
        "model_id": repo, "immutable_commit": full_revision,
        "model_kind": "random_pretrained_checkpoint_for_debugging_only",
        "device": "cpu", "dtype": "float32", "seed": 0,
        "cases": cases, "environment": environment_snapshot(),
        "vllm_gpu_baseline": "not_run", "performance_claim": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Stage 01 Qwen3 real-weight forward/token oracle PASS: {full_revision}")


if __name__ == "__main__":
    main()
