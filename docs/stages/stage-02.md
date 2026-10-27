# Stage 02 — Paged KV physical storage and ownership

**Status:** Stage 02 CPU physical-paging correctness verified in GitHub Actions. **Stage 03 not started.** GPU PagedAttention is outside Stage 02.

## Components

- [x] CPU-only `BlockManager`: fixed physical capacity, free list, immutable generation-stamped handles and per-request Block Table.
- [x] Transactional `begin/commit/abort/release` and conservative COW for partial last pages.
- [x] Atomic OOM handling, stale/replayed reservation rejection, lock-protected allocator operations and invariant oracle.
- [x] `PagedKVStore`: physical K/V tensors, shape/dtype/device validation, page writes and dense readback.
- [x] Optional `LLMEngine(paged_cache=...)` path; unpaged reference remains default; `finally` cleanup.
- [x] Randomized allocation, injected physical write faults, stale lease, OOM, threading and two-model token parity tests authored.
- [x] CI + real pinned HF checkpoints passed; [reference run 40484347235](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40484347235) and [full static/test CI 40484347245](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40484347245).
- [x] CPU scoped acceptance and model/hardware matrix documented. Trained Qwen3-0.6B and vLLM GPU benchmarks are not run.

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

**Measured verification:** 21 Stage 02 tests passed, and existing 22 Stage 01 model tests passed. The actual pinned `sshleifer/tiny-gpt2` and `llamafactory/tiny-random-qwen3` generated IDs matched the independent dense and paged paths; the second checkpoint contains random debugging weights, not production language capability.

**Performance boundary:** no PagedAttention kernel. The reference path materializes full dense KV and checks committed prefix immutability each append; memory/performance conclusions require a separate controlled experiment.

See [transaction ADR](../adr/0004-paged-kv-transaction.md). Future Stage 03 may add continuous batching, not now.
