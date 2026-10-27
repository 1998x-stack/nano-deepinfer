# Acceptance and release quality gates

## Stage 00 exit gates

- [x] Three upstream full SHAs frozen in `docs/sources.lock.json`.
- [x] Architecture/source mapping, decision records and explicit interfaces authored.
- [x] Reproducible CPU synthetic smoke and environment schema.
- [x] GitHub Actions CI passed on initial Stage 00 merged revision; [main workflow](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40467149468).
- [ ] HF actual small Dense pretrained-checkpoint baseline executed (model/tokenizer SHA pinned). **NOT RUN**.
- [ ] Pinned upstream vLLM real inference benchmark in compatible environment. **NOT RUN**.
- [ ] Stage 00 strict closeout approved with missing external baselines resolved or documented exception.

**Current strict gate:** PARTIAL, not PASS. Stage 01 has **not** started. CI and synthetic tests do not establish inference correctness.

## Global Definition of Done

- [ ] Tests for each implemented function, CPU oracle and precise failure semantics.
- [ ] Ruff, strict MyPy, Pytest, reproducible CI and changes reviewed.
- [ ] Logits/greedy token exactness tested on same model revision, dtype/tokenizer and backend equivalence when Stage 01 exists.
- [ ] Cancellation, OOM, double release, async results and rollback tested when mechanisms exist.
- [ ] Benchmarks record checkpoint/toolchain commit, warmup, repeats, actual resource inventory and failures.
- [ ] No source copying without LICENSE/NOTICE audit; honest capability README.
- [ ] Single-step fallback always available; observed regressions reported.

## Release gates

- `v0.1`: Stages 00–04 with real single-node Dense model and scheduler/cache correctness.
- `v0.2`: Stages 05–06 with optimized/reference ablation and async safety evidence.
- `v0.3`: Stages 07–09 with reference-valid experiments; GPU-demanding algorithms may remain experimental.
- `v1.0`: Stages 10 and 12 required stable single-node API, load/fault/security docs; Stage 11 optional experimental.
