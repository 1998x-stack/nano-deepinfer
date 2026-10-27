# Stage 02 — Paged KV physical storage and ownership

**Status:** implementation in review; completion is conditioned on CI and model-reference validation. **Stage 03 not started.**

## Components

- [x] CPU-only `BlockManager`: fixed physical capacity, free list, immutable generation-stamped handles and per-request Block Table.
- [x] Transactional `begin/commit/abort/release` and conservative COW for partial last pages.
- [x] Atomic OOM handling, stale/replayed reservation rejection, lock-protected allocator operations and invariant oracle.
- [x] `PagedKVStore`: physical K/V tensors, shape/dtype/device validation, page writes and dense readback.
- [x] Optional `LLMEngine(paged_cache=...)` path; unpaged reference remains default; `finally` cleanup.
- [x] Randomized allocation, injected physical write faults, stale lease, OOM, threading and two-model token parity tests authored.
- [ ] CI + real pinned HF checkpoints on committed SHA have passed and evidence recorded.
- [ ] Release-readiness review and model hardware matrix documented.

## Strict exit acceptance

1. Tests prove no double allocation, stale-handle alias, lost capacity or cache publication after failed reserve/write.
2. GPT-2/Qwen3 outputs from physical paging are token-exact against same-weights unpaged reference; actual pinned HF baseline run, not guessed.
3. `BlockManager.assert_consistent()` passes after randomized operations and after teardown.
4. Normal finish, early-stop, OOM, cancellation-equivalent release and write failure reclaim all physical pages.
5. No claim of throughput improvement or PagedAttention; full Qwen3-0.6B and vLLM GPU baseline remain explicitly NOT RUN.

## Commands

```bash
pip install -e '.[dev,reference]'
pytest -q tests/test_stage02_block_manager.py tests/test_stage02_paged_kv.py
python examples/stage02_paged_reference.py
python scripts/stage02_real_model_parity.py
```

See [transaction ADR](../adr/0004-paged-kv-transaction.md). Future Stage 03 may add continuous batching, not now.
