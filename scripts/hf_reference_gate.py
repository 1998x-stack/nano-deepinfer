"""Stage 00 external checkpoint oracle: HF full-prefix versus KV-cached decoding.

This measures *Transformers*, not nano-deepinfer. It does not claim vLLM parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from nano_deepinfer.foundation import environment_snapshot

MODEL_ID = "sshleifer/tiny-gpt2"
REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("max-new-tokens must be between 1 and 64")

    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    torch.set_num_threads(2)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()
    prompt = "A small language model stores cached attention keys and values."
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated: list[int] = []
    max_abs_logit_error = 0.0
    duration_start = perf_counter()

    with torch.inference_mode():
        cached = model(input_ids=input_ids, use_cache=True)
        full = model(input_ids=input_ids, use_cache=False)
        err = (cached.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()
        max_abs_logit_error = max(max_abs_logit_error, err)
        if err > 1e-4:
            raise AssertionError(f"prefill logits differ: {err}")
        past = cached.past_key_values
        for _ in range(args.max_new_tokens):
            predicted = torch.argmax(full.logits[:, -1, :], dim=-1, keepdim=True)
            next_token = int(predicted.item())
            generated.append(next_token)
            input_ids = torch.cat((input_ids, predicted), dim=1)
            full = model(input_ids=input_ids, use_cache=False)
            cached = model(input_ids=predicted, past_key_values=past, use_cache=True)
            past = cached.past_key_values
            err = (cached.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()
            max_abs_logit_error = max(max_abs_logit_error, err)
            if err > 1e-4:
                raise AssertionError(f"decode logits differ: {err}")
    elapsed = perf_counter() - duration_start

    record = {
        "schema_version": 1,
        "stage": "00",
        "kind": "hf_pretrained_checkpoint_prefill_decode_reference",
        "status": "passed",
        "engine": "huggingface-transformers-only",
        "model_id": MODEL_ID,
        "revision_sha": REVISION,
        "dtype": "float32",
        "device": "cpu",
        "prompt_token_ids": input_ids.shape[1] - len(generated),
        "generated_token_ids": generated,
        "max_abs_logit_error": max_abs_logit_error,
        "max_new_tokens": args.max_new_tokens,
        "wall_seconds": elapsed,
        "environment": environment_snapshot(),
        "nano_deepinfer_parity": "not_tested_until_stage01",
        "vllm_parity": "not_tested_no_gpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("HF pretrained reference gate PASS:", args.output)


if __name__ == "__main__":
    main()
