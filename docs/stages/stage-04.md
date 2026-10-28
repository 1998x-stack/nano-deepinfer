# Stage 04 — Chunked Prefill and shared prefix cache

**Status:** Code implemented in `feat/stage-04-prefix-cache`; model-reference CI / review is the merge gate.

## Deliverables

- [x] Optional full-block-only content-addressed immutable prefix index with namespace fields: model weights, tokenizer, positional encoding, tenant, dtype, adapter and schema.
- [x] Physical KV page **sharing** with cache-owned pins, refcount tracking, generation-safe handles and explicit `attach_prefix`; no deep-copy of cached K/V payload on hit.
- [x] Last-logits reuse for exact prompt hits; partial-prefix re-use when an aligned cached prompt begins a longer prompt.
- [x] Per-tenant invalidation, LRU eviction, bounded cache-pin capacity and reclaim under KV allocation pressure.
- [x] Conservative copy-on-write and Stage 02 transactional rollback retained.
- [x] Configurable `max_prefill_chunk_size` limit within the unified token budget; Stage 03 logic for mixed prefill/decode preserved.
- [x] CPU tests for namespace isolation, stale and shared handles, pinned physical ownership, COW, evicted-but-live readers, chunk size and GPT-2/Qwen3 cached-versus-reference Token parity.
- [x] Pinned public GPT-2 pretrained and tiny random Qwen3 reference script with actual raw CPU latency repetitions, before/after hit counters and full token ID oracle.
- [ ] CI and real model run green on final PR SHA with raw JSON artifact archived.
- [ ] Review and merge to main; milestone branch pointers fast-forwarded.

## Strict CPU correctness acceptance

1. `assert_consistent()` passes with simultaneously cached and attached physical pages; `PoolStats.used` counts physical occupied blocks, not sum of refs.
2. No namespace cross-tenant or model/adapter/tokenizer/position/dtype cross-hit; same tokens alone do not authorize cache reuse.
3. Pinned entries survive producer request completion; eviction retains live-reader pages; release + cache invalidation restores all capacity.
4. Prefill chunk cap and step token budget are respected; full-cache hits reuse cached last logits and bypass prompt model forward.
5. Three controlled tiny-model CPU runs per baseline/cached condition, raw artifacts and exact generated ID parity with immutable Hugging Face checkpoints.
6. No unsupported GPU PagedAttention/throughput claims. Uncached path remains an explicit feature-flag fallback.

## Commands

```bash
python -m pip install -e ".[dev,reference]"
pytest -q tests/test_stage04_prefix_index.py tests/test_stage04_prefix_runtime.py
python scripts/stage04_real_model_parity.py
```

## Deliberate deferrals

No fused GPU tensor batching, partial-block prefix caching, general attention masks, speculative state sharing or async KV refs. Existing vLLM same-hardware benchmark and Qwen3-0.6B trained weights remain unresolved original stage gates. See [ADR 0006](../adr/0006-shared-prefix.md).
