# mypy: ignore-errors
"""Stage 03 pinned real-checkpoint multi-request correctness, not a performance claim."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from nano_deepinfer.batch_engine import ContinuousBatchEngine
from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.dense_gpt2 import ReferenceGPT2
from nano_deepinfer.dense_qwen3 import ReferenceQwen3
from nano_deepinfer.foundation import SamplingParams, environment_snapshot


def main() -> None:
    torch.set_num_threads(2)
    checkpoints = (
        ("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be", "gpt2"),
        ("llamafactory/tiny-random-qwen3", "81d6f5f5e05ed53ea8a1d19431266a486e46bbd8", "qwen3"),
    )
    cases = []
    for model_id, revision, family in checkpoints:
        hf = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False,
            torch_dtype=torch.float32,
        ).eval()
        model = ReferenceGPT2.from_hf_gpt2(hf) if family == "gpt2" else ReferenceQwen3.from_hf_qwen3(hf)
        cfg = model.config
        heads = cfg.num_heads if family == "gpt2" else cfg.num_kv_heads
        dim = cfg.hidden_size // cfg.num_heads if family == "gpt2" else cfg.head_dim
        cache = PagedKVStore(16, 4, cfg.num_layers, heads, dim)
        engine = ContinuousBatchEngine(
            model, cache, max_num_scheduled_tokens=7, max_num_seqs=3,
            model_fingerprint=revision,
        )
        prompts = {"one": (1, 2, 3), "two": (4, 5), "three": (2, 7, 5, 8)}
        sampling = SamplingParams(max_new_tokens=5, seed=0)
        expected: dict[str, tuple[int, ...]] = {}
        with torch.inference_mode():
            for rid, prompt in prompts.items():
                input_ids = torch.tensor([prompt], dtype=torch.long)
                ref_ids = hf.generate(
                    input_ids=input_ids, max_new_tokens=5, do_sample=False,
                    pad_token_id=0, eos_token_id=None,
                )
                expected[rid] = tuple(ref_ids[0, len(prompt):].tolist())
                engine.add_request(rid, prompt, sampling)
            observed = engine.run(max_steps=200)
        for rid, want in expected.items():
            if observed[rid].generated_token_ids != want:
                raise AssertionError(f"{family}: {rid}: incorrect generated IDs")
        cache.assert_consistent()
        if cache.manager.stats().used:
            raise AssertionError("KV leaked after draining batch")
        if not any(
            {event.kind for event in step.events} >= {"prefill", "decode"}
            for step in engine.history
        ):
            raise AssertionError("expected mixed prefill/decode batch not observed")
        cases.append({
            "model_id": model_id, "revision_sha": revision, "family": family,
            "requests": len(prompts), "generated_token_ids": {
                rid: list(result.generated_token_ids) for rid, result in observed.items()
            },
            "step_count": len(engine.history),
            "max_token_budget": 7,
            "max_num_seqs": 3,
            "preemptions": engine.preemption_count,
            "cache_live_blocks_after": cache.manager.stats().used,
        })
    dest = Path("benchmarks/results/stage03-real-model-cpu.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "schema_version": 1, "stage": "03", "status": "passed",
        "kind": "logical_multi_request_serial_model_execution",
        "dtype": "float32", "device": "cpu", "cases": cases,
        "environment": environment_snapshot(),
        "gpu_batch_acceleration": "not_implemented",
        "vllm_benchmark": "not_run",
        "runtime_performance_comparison": "not_run",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Stage 03 real-checkpoint scheduler parity PASS:", dest)


if __name__ == "__main__":
    main()
