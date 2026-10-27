# mypy: ignore-errors
"""Record correctness + latency for actual pinned HF checkpoint, no performance claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from nano_deepinfer.foundation import SamplingParams, environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    from nano_deepinfer.dense_gpt2 import ReferenceGPT2
    from nano_deepinfer.engine import LLMEngine

    model_id = "sshleifer/tiny-gpt2"
    revision = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
    torch.set_num_threads(2)
    reference = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, torch_dtype=torch.float32
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    nano = ReferenceGPT2.from_hf_gpt2(reference)
    prompts = ["Incremental inference", "Language modeling is"]
    cases = []
    for prompt in prompts:
        ids = tuple(tokenizer.encode(prompt))
        params = SamplingParams(max_new_tokens=8, seed=0)
        with torch.inference_mode():
            hf_ids = torch.tensor([ids], dtype=torch.long)
            hf_outputs = reference.generate(
                hf_ids, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            start = perf_counter()
            result = LLMEngine(nano, model_fingerprint=revision,
                               eos_token_id=tokenizer.eos_token_id).generate(ids, params)
            elapsed = perf_counter() - start
            expected = tuple(hf_outputs[0, len(ids):].tolist())
        if result.generated_token_ids != expected:
            raise AssertionError(f"nano/HF output mismatch: {result.generated_token_ids} vs {expected}")
        cases.append({
            "prompt_length": len(ids), "output_length": len(result.generated_token_ids),
            "nano_generated_ids": list(result.generated_token_ids),
            "hf_generated_ids": list(expected),
            "nano_wall_seconds": elapsed, "greedy_token_exact": True
        })
    record = {
        "schema_version": 1, "stage": "01",
        "engine": "nano-deepinfer-independent-gpt2-reference",
        "model_id": model_id, "revision_sha": revision, "device": "cpu",
        "precision": "float32", "status": "passed", "cases": cases,
        "metrics_note": "Unwarmed one-shot CPU wall-time; NOT comparable to vLLM, not TTFT/TPOT.",
        "environment": environment_snapshot(),
        "vllm_gpu_benchmark": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Stage 01 HF token parity PASS:", args.output)


if __name__ == "__main__":
    main()
