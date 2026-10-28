# mypy: ignore-errors
"""Pinned checkpoint prefix-cache correctness and repeatable CPU measurements.

Results report *this* run's small-model latency, not general speedup claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM

from nano_deepinfer.batch_engine import ContinuousBatchEngine
from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.cache.prefix_cache import CacheNamespace
from nano_deepinfer.dense_gpt2 import ReferenceGPT2
from nano_deepinfer.dense_qwen3 import ReferenceQwen3
from nano_deepinfer.foundation import SamplingParams, environment_snapshot


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cases = []
    checkpoints = (
        ("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "gpt2"),
        ("llamafactory/tiny-random-qwen3", "81d6f5f5e05ed53ea8a1d19431266a486e46bbd8", "qwen3"),
    )
    for model_id, sha, family in checkpoints:
        hf = AutoModelForCausalLM.from_pretrained(
            model_id, revision=sha, trust_remote_code=False, torch_dtype=torch.float32
        ).eval()
        model = (
            ReferenceGPT2.from_hf_gpt2(hf)
            if family == "gpt2" else ReferenceQwen3.from_hf_qwen3(hf)
        )
        cfg = model.config
        heads = cfg.num_heads if family == "gpt2" else cfg.num_kv_heads
        dim = cfg.hidden_size // cfg.num_heads if family == "gpt2" else cfg.head_dim
        scope = CacheNamespace(sha, sha, f"{family}-default-position", "bench-tenant", "torch.float32")
        cache = PagedKVStore(20, 4, cfg.num_layers, heads, dim, max_cached_blocks=4)
        no_cache = PagedKVStore(20, 4, cfg.num_layers, heads, dim)
        cached_engine = ContinuousBatchEngine(
            model, cache, prefix_namespace=scope, model_fingerprint=sha,
            max_num_scheduled_tokens=8, max_prefill_chunk_size=4,
        )
        baseline = ContinuousBatchEngine(
            model, no_cache, model_fingerprint=sha, max_num_scheduled_tokens=8,
            max_prefill_chunk_size=4,
        )
        params = SamplingParams(max_new_tokens=4, temperature=0.0, seed=0)
        warm_prompt = (1, 2, 3, 4)
        suffix_prompt = (1, 2, 3, 4, 5, 6)
        with torch.inference_mode():
            ids = torch.tensor([suffix_prompt], dtype=torch.long)
            hf_ids = hf.generate(
                input_ids=ids, max_new_tokens=4, do_sample=False,
                eos_token_id=None, pad_token_id=0,
            )[0, len(suffix_prompt):].tolist()
            # Warm the same model and its two cache configurations identically.
            for engine in (baseline, cached_engine):
                engine.add_request("warm", warm_prompt, params)
                engine.run()
            if cache.prefix_cache is None or cache.prefix_cache.entries != 1:
                raise AssertionError("aligned warm prompt was not cached")
            elapsed = {"baseline": [], "cached": []}
            for i in range(3):
                for label, engine in (("baseline", baseline), ("cached", cached_engine)):
                    rid = f"repeat-{i}"
                    start = perf_counter()
                    engine.add_request(rid, suffix_prompt, params)
                    produced = engine.run()[rid].generated_token_ids
                    seconds = perf_counter() - start
                    elapsed[label].append(seconds)
                    if list(produced) != hf_ids:
                        raise AssertionError(f"{family} {label}: generated ID mismatch")
            cache.assert_consistent()
            no_cache.assert_consistent()
            if cached_engine.prefix_hits < 3 or cached_engine.reused_prefix_tokens < 12:
                raise AssertionError("prefix hit and reused-token accounting mismatch")
            if no_cache.manager.stats().used != 0:
                raise AssertionError("uncached engine leaked KV pages")
            before_clear = cache.manager.stats().used
            cache.clear_prefix()
            if cache.manager.stats().used != 0:
                raise AssertionError("prefix pin leak after explicit invalidation")
        cases.append({
            "model_id": model_id, "revision_sha": sha,
            "family": family, "warm_prompt_tokens": 4,
            "target_prompt_tokens": 6, "output_tokens": 4,
            "reference_generated_ids": hf_ids,
            "prefix_hits": cached_engine.prefix_hits,
            "reused_prefix_tokens": cached_engine.reused_prefix_tokens,
            "pinned_blocks_before_clear": before_clear,
            "blocks_after_clear": cache.manager.stats().used,
            "baseline_wall_seconds": elapsed["baseline"],
            "cached_wall_seconds": elapsed["cached"],
            "baseline_median_seconds": median(elapsed["baseline"]),
            "cached_median_seconds": median(elapsed["cached"]),
        })
    artifact = Path("benchmarks/results/stage04-real-model-cpu.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "schema_version": 1, "stage": "04", "status": "passed",
        "environment": environment_snapshot(),
        "device": "cpu", "dtype": "float32", "repeats": 3,
        "cases": cases, "vllm_gpu_comparison": "not_run",
        "full_qwen3_0_6b": "not_run",
        "scope": "small CPU checkpoint illustration; no generalized speedup claim",
        "model_calls_fused": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Stage 04 pinned model prefix reuse + HF parity PASS:", artifact)


if __name__ == "__main__":
    main()
