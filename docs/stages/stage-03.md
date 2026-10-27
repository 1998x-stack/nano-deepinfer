# Stage 03 — Unified token-budget scheduling (CPU reference)

**Status:** Implemented, awaiting final CI evidence and merge. **Stage 04 not started.**

## Delivery

- [x] Independent `UnifiedScheduler`: bounded `max_num_scheduled_tokens`, `max_num_seqs`, FCFS initial queue and residency-aware round-robin; unique immutable `ScheduledWork` per request per step with version/input offset
- [x] A minimal prompt chunk fallback, including full recomputation of evicted requests; optimized Chunked Prefill policy and prefix sharing are reserved for Stage 04
- [x] `ContinuousBatchEngine` allowing request arrival between synchronous steps, interleaved Prefill/Decode, EOS, stop-token IDs and seeded sampling
- [x] Physical KV block capacity coupled to Stage 02 `PagedKVStore`; deterministic victim eviction, request-version fencing, lossless regeneration of cached forward state
- [x] Cancellation, per-request failure isolation, bounded admission/backpressure, terminal results, zero-leak cleanup and step history
- [x] Scheduler tests for 100-request fairness and exact token budgets; real Dense GPT-2/Qwen3 tests for 100-request completion, late arrival, cancellation, OOM, preemption, crash isolation and serial oracle token equality
- [x] Pinned tiny GPT-2 pretrained and tiny random Qwen3 real checkpoints included in CPU HF workflow
- [ ] CI results verified on final code SHA, raw evidence recorded and PR merged

## Acceptance

1. No request scheduled twice per planning step; total planned tokens never exceed configured budget
2. Continuous admission while existing requests decode; workload drains without lost or duplicate generated Token IDs
3. Low-capacity preemption evicts physical pages and recomputes committed tokens without changing visible generations; stale version-fenced work does not commit
4. Canceled, failed and finished requests reclaim physical pages; other requests can continue after one fails
5. Per-request Greedy/seeded outputs equal independent sequential oracle; real pinned HF reference generated IDs match
6. Never describe this as fused GPU tensor batching: per-request model forward is **serial** and attention still materializes contiguous KV

## Constraints and research debt

Synchronous process only. Cancellation between steps, not during a running kernel. Residency preference under constrained cache reduces thrashing but may increase latency for newly arrived requests; no guaranteed real-time fairness. GPU vLLM comparison, Qwen3-0.6B trained weights and speedup benchmarks remain **NOT RUN**. Stage 04 (prefix sharing, optimized chunks) and Stage 05 (GPU kernels) not implemented.

## Reproduction

```bash
pip install -e ".[dev,reference]"
pytest -q tests/test_stage03_scheduler.py tests/test_stage03_batch_engine.py
python scripts/stage03_real_model_parity.py
```

See [ADR 0005](../adr/0005-unified-scheduler.md) and [benchmark plan](../benchmarks.md).
