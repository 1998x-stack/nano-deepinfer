# ADR 0005 — Unified scheduling separated from Dense execution

**Date:** 2026-10-27. **Status:** Stage 03 research implementation, pending final CI.

## Decision

Preserve an independent CPU-only control plane `UnifiedScheduler` for request ordering, token-budget accounting and planning. Its immutable `ScheduledBatch` groups `ScheduledWork` items carrying step ID, request version, input offset and operation kind (prefill/decode). Strict maximum scheduled tokens and maximum distinct requests are enforced per step. FCFS initial order with residency-aware round-robin favors requests already holding GPU/CPU KV blocks so that admitting many new prompts does not evict every existing decode.

An independently implemented `ContinuousBatchEngine` executes the logical work items **serially** (one model sequence per Torch forward). The model's output and sampling are independent of which other requests were serviced before it; each request owns a separate seed/generator. New requests can be submitted between steps.

## KV memory pressure

Before publishing full new per-layer KV, `PagedKVStore` reserves its required physical pages transactionally (Stage 02). On `KVCapacityError`, the executor may evict a deterministic resident victim, increment the victim version and discard its cache; already-generated output tokens stay preserved. The victim later recomputes prompt + published generation in bounded prefill chunks. The same step ignores stale selected work after a version changes. If no victim can free capacity, fail only the current request and reclaim all its pages.

## Error and cancellation policy

Finish/cancel/error is terminal; per-request state remains for inspection and duplicate ID rejection. Synchronous sampling updates become externally visible only after any required KV forward and publish succeeds. On a KV publish failure, sampled RNG state is restored and the failing request ends with `failed` status. Cancellation is idempotent and guaranteed between steps, not midway through GPU execution.

## Alternatives and trade-offs

Do not introduce vLLM runtime as scheduler, to preserve independence. Do not call this GPU batching: current model accepts batch size one and materializes dense KV. Preemption/recomputation may be costly; future Stage 04 adds prefix reuse and optimized chunked prefill. Fused model kernels, true GPU multi-sequence batches and asynchronous step/commit synchronization remain future work.

## Verification

Seeded, multi-request Oracle tests, 100-request scheduling and CPU model completion, mixed phases, cancellation, OOM pressure and crash injection; two pinned real-model HF references. No unmeasured token/s performance claims or complete Qwen3-0.6B coverage.
