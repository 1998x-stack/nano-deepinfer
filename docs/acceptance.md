# Acceptance and release quality gates

## Stage 00 exit gates

- [x] Repository research snapshots pinned by full SHA in `docs/sources.lock.json`.
- [x] Architecture, provenance mapping, decisions and explicit interface contracts authored.
- [x] Reproducible synthetic CPU smoke script and environment schema authored.
- [ ] CPU smoke and tests executed in GitHub Actions on committed SHA and PASS is recorded.
- [ ] Hugging Face reference baseline executed with actual model checkpoint. **Blocked until transformers/checkpoint are available.**
- [ ] Independent vLLM baseline executed on compatible GPU. **Blocked pending usable GPU stack.**
- [ ] Owner reviews CI results and accepts Stage 00 scope.

Stage 00 can be merged as a documented **foundation** even if hardware baselines are deferred, but its strict research exit gate must be marked `PARTIAL`, not falsely marked PASS. Starting Stage 01 should explicitly acknowledge this prerequisite gap.

## Global DoD

- [ ] Source checks: Ruff and MyPy; package import and unit pytest.
- [ ] Deterministic reference correctness per supported model and dtype.
- [ ] Reliable handling of cancellation, timeout, errors, reservation cleanup when stage applies.
- [ ] Benchmark artifacts carry model weights SHA, tokenizer SHA, git SHA, seeds, hardware, software, warmup and workload.
- [ ] Performance regressions reported, not suppressed; optimized code guarded by fallback.
- [ ] Documentation, compatibility matrix, dependencies and upstream attribution updated.
- [ ] CI and branch history verified after merge.

## Release gates

`v0.1`: Stage 01–04 dense, cache, batching, prefix correctness. `v0.2`: Stage 05–06 reproducible async + optimized vs eager evidence. `v0.3`: Stage 07–09 reference-valid experiments (hardware-gated features explicitly optional). `v1.0`: Stage 10+12 stable single-node API, security/reliability/performance validation; Stage 11 P/D optional.

