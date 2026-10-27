"""Optional Stage 00 Transformers baseline. Requires a real immutable model revision.

This command is NOT run in the default CPU-only CI; output files record real runs only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

from nano_deepinfer.foundation import environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--revision", required=True, help="40-character immutable commit SHA")
    parser.add_argument("--prompt", default="Explain the role of a KV cache in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character git SHA")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False, dtype=torch.float32
    ).eval().to("cpu")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        start = perf_counter()
        ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        duration = perf_counter() - start
    record = {
        "schema_version": 1,
        "stage": "00",
        "kind": "transformers_real_model_greedy_baseline",
        "status": "measured",
        "model": args.model,
        "revision_sha": args.revision,
        "dtype": "float32",
        "device": "cpu",
        "temperature": 0.0,
        "seed": 0,
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "generated_ids": ids[0].tolist(),
        "duration_seconds": duration,
        "environment": environment_snapshot(),
        "comparison_with_nano_deepinfer": "not_run_until_stage01",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"HF real-model baseline saved: {args.output}")


if __name__ == "__main__":
    main()
