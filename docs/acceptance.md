# Acceptance and release quality gates

## Stage 00: design complete, strict exit PARTIAL
- [x] Upstream commits locked, architecture and RFC, source research, CPU synthetic tests and CI.
- [x] Pinned genuine pretrained HF tiny GPT-2 CPU reference measured: [run 40468544667](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667).
- [ ] Matched **vLLM GPU** reference performance benchmark: **NOT RUN**. [ADR 0003](adr/0003-stage-00-gpu-deferral.md) defers this required validation to the Stage 05 optimization prerequisite.

## Stage 01: core PASS, complete hardware/model exit OPEN
- [x] Independent GPT-2 and Qwen3 Dense model forward including separate prefill/cache decode; only batch size one, eager/unpaged KV.
- [x] Greedy, seeded temperature, top-k, top-p, eos and stop token IDs.
- [x] Tiny pretrained GPT-2 HF logits/generated IDs match; small Qwen3 HF and immutable random-weight checkpoint full/cache/logit/greedy parity: [run 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422), [model tests 22 PASS](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40476699366).
- [x] Reference CI, example, reproducible source/model locks and honest README status.
- [ ] Complete pretrained **Qwen/Qwen3-0.6B** tokenizer/weight comparison on target hardware.
- [ ] Robust cancellation and device OOM recovery under failure injection.

The validated Stage 01 reference code is merged into `main`. The above unverified criteria are **not** considered PASS.

## Stage 02: CPU physical-paging correctness PASS

- [x] Independently authored `BlockManager` with generation-stamped handles, finite block pool, request Block Tables and guarded reservation lifecycle.
- [x] Transactional `begin/commit/abort/release`, conservative COW, stale-handle and replay rejection, deterministic seeded randomized allocations and threaded tests.
- [x] Physical page tensor store `PagedKVStore`, committed-prefix immutability checks, explicit dtype/shape/device validation, injected mid-write failure and OOM cleanup.
- [x] GPT-2 and Qwen3 token-exact output parity, preserving the Stage 01 default reference: [real checkpoint run 40484347235](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40484347235) (Stage 02: 21 focused tests and two pinned HF weights).
- [x] No claims of PagedAttention/speedups; see [ADR 0004](adr/0004-paged-kv-transaction.md).
- [ ] Direct GPU PagedAttention and asynchronous write fencing: explicitly **OUT OF STAGE 02**.
- [ ] Matching-hardware vLLM performance baseline remains **NOT RUN**; full Qwen3-0.6B trained checkpoint remains **NOT RUN**.

This passes the **bounded Stage 02 CPU correctness gate**; it does not close earlier hardware-specific gates or release v0.1. Stage 03 CPU request-interleaving correctness is now verified.

## Stage 03: CPU logical scheduling correctness PASS

- [x] Deterministic unified token-budget scheduler with FCFS admission, bounded round-robin, per-step request versions and mixed prefill/decode.
- [x] Multi-request runner with Stage 02 physical block pressure, eviction, recompute, cancel, fault isolation and admission backpressure.
- [x] CPU 100-request property/real-model tests authored and actual two-checkpoint HF parity workflow configured.
- [x] 17 focused Stage 03 tests and pinned two-model HF greedy parity passed: [reference run 40489283022](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40489283022), [CI 40489283080](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40489283080); raw correctness artifact uploaded.
- [x] [PR #8](https://github.com/1998x-stack/nano-deepinfer/pull/8) merged after green CPU CI; `main`, `v0.1`, `v0.2`, `v0.3`, `v1.0` updated to `5dfa0ef4a89d2645b588dd4179333f40bfa9ac43`. These are planning branches, **not** release tags.

**Precise capability:** multi-request logical batching only. Model forwards remain serial per request, no GPU throughput claim. Stage 04 not begun. The Stage 00 vLLM GPU and full trained Qwen3-0.6B gates remain open.

## Common DoD
Tests, correctness oracle, SHA-pinned weights and workload, Ruff/MyPy/Pytest, explicit failures and source notices. Never claim vLLM performance without same hardware/loads and reproducible results.

## Release gates
`v0.1`: Stage 00–04 dense/paged-KV/batching/prefix correctness.
`v0.2`: Stage 05–06 GPU optimized/async reproducibility.
`v0.3`: Stage 07–09 MoE/speculation/SWA experimental correctness.
`v1.0`: Stage 10/12 stable serving/release; Stage 11 optional research.
