# mypy: ignore-errors
"""Actual pinned GPT-2/Qwen3 checkpoint correctness for the Stage 02 paging adapter."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.dense_gpt2 import ReferenceGPT2
from nano_deepinfer.dense_qwen3 import ReferenceQwen3
from nano_deepinfer.engine import LLMEngine
from nano_deepinfer.foundation import SamplingParams, environment_snapshot


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(2)
    models = (
        ("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "gpt2"),
        ("llamafactory/tiny-random-qwen3", "81d6f5f5e05ed53ea8a1d19431266a486e46bbd8", "qwen3"),
    )
    cases = []
    for identifier, revision, family in models:
        hf = AutoModelForCausalLM.from_pretrained(
            identifier, revision=revision, trust_remote_code=False, torch_dtype=torch.float32
        ).eval()
        independent = (
            ReferenceGPT2.from_hf_gpt2(hf) if family == "gpt2"
            else ReferenceQwen3.from_hf_qwen3(hf)
        )
        cfg = independent.config
        if family == "gpt2":
            heads = cfg.num_heads
            dim = cfg.hidden_size // cfg.num_heads
        else:
            heads, dim = cfg.num_kv_heads, cfg.head_dim
        pool = PagedKVStore(20, 4, cfg.num_layers, heads, dim)
        tokens = (1, 2, 3, 4, 5)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.inference_mode():
            reference_logits = hf(input_ids=input_ids, use_cache=False).logits
            own_logits, dense_kv = independent(input_ids)
            delta = float((reference_logits - own_logits).abs().max().item())
            if delta > 0.002:
                raise AssertionError(f"model logits mismatch: {family} {delta}")
            pool.write("probe", dense_kv)
            restored = pool.read("probe")
            for expected, actual in zip(dense_kv, restored):
                for left, right in zip(expected, actual):
                    if not torch.equal(left, right):
                        raise AssertionError("paged KV roundtrip mismatch")
            pool.release("probe")
            params = SamplingParams(max_new_tokens=6, temperature=0, seed=7)
            hf_ids = hf.generate(
                input_ids=input_ids, do_sample=False, max_new_tokens=6,
                pad_token_id=0, eos_token_id=None,
            )[0, len(tokens):].tolist()
            direct = LLMEngine(independent).generate(tokens, params).generated_token_ids
            paged = LLMEngine(independent, paged_cache=pool).generate(
                tokens, params
            ).generated_token_ids
        if list(paged) != hf_ids or direct != paged:
            raise AssertionError(f"full/paged/HF token IDs disagree: {family}")
        pool.assert_consistent()
        if pool.manager.stats().used or pool.manager.stats().owners:
            raise AssertionError("page leak")
        cases.append({
            "family": family,
            "model_id": identifier,
            "revision_sha": revision,
            "full_logits_abs_error": delta,
            "hf_generated_ids": hf_ids,
            "dense_generated_ids": list(direct),
            "paged_generated_ids": list(paged),
            "reserved_physical_bytes": pool.reserved_bytes,
            "live_blocks_after_generation": pool.manager.stats().used,
        })
    out = Path("benchmarks/results/stage02-real-model-cpu.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": "02", "status": "passed", "schema_version": 1,
        "engine": "independent-dense-attention-with-paged-kv-storage",
        "hardware": "CPU", "dtype": "float32",
        "cases": cases, "environment": environment_snapshot(),
        "latency_comparison": "not_run",
        "gpu_vllm_comparison": "not_run",
        "notes": "correctness only; dense materialization is not PagedAttention",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Stage 02 real model correctness PASS:", out)


if __name__ == "__main__":
    main()
