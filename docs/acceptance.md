# Acceptance and release quality gates

## Stage 00 exit gates (2026-10-27)

- [x] Three upstream full SHAs frozen in `docs/sources.lock.json`.
- [x] Source research, architecture and interface contracts authored.
- [x] CPU-only synthetic smoke and reproducible environment schema.
- [x] Stage 00 CI green on Python 3.11/3.12.
- [x] **Actual pinned HF pretrained GPT-2 reference executed** (model `sshleifer/tiny-gpt2@5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`). Full and cached HF logits align; [successful run](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667), raw JSON artifact stored there.
- [ ] **Real GPU vLLM reference benchmark still not run**. No compatible GPU environment linked for this execution.
- [ ] Owner reviews and closes strict Stage 00 acceptance, or explicitly documents a limited exception for the missing GPU comparison.

**Status: PARTIAL, not PASS.** HF full-vs-cached self-comparison does not establish independent nano engine correctness. Stage 01 prototype may be developed on a PR branch, but must not be declared released or merged until gate resolution.

## Global Definition of Done

- [ ] New features have independently executable unit, correctness and failure-mode tests.
- [ ] Ruff, MyPy and Pytest green in the relevant dependency environment.
- [ ] HF equivalence uses same checkpoint, tokenizer, dtype and documented tolerance.
- [ ] CPU/GPU timing reports capture environmental constraints, warmup, latency distribution and failures.
- [ ] Cache and async operations have explicit ownership/rollback invariants when implemented.
- [ ] Source licenses and notices maintained; README capability claims match actual code.
- [ ] CI and branch statuses rechecked after merge.

## Release gates

`v0.1`: Stage 00–04 correctness including a real supported Dense model, paging and batching.

`v0.2`: Stage 05–06 optimized and asynchronous paths with independent reference fallback and A/B evidence.

`v0.3`: Stage 07–09 experimental MoE/speculation/SWA validated against their oracles.

`v1.0`: Stage 10 and 12 reliable single-node service with reproducible release record; Stage 11 optional.
