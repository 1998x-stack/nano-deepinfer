# Stage 00 — Research & reproducible engineering foundation

**Status: in progress; no actual LLM performance measured.**  **Estimated effort:** 1–2 weeks. **Prerequisites:** none.

## Work packages

- [x] S00-01: Freeze nano-vLLM, vLLM V1 and DeepInfer commit SHAs; source/license table and research mapping.
- [x] S00-02: Research scheduling, KV block, model runner, C++ async boundaries and explicitly identify what is not open-sourced.
- [x] S00-03: Architecture RFC, model adapter interface, request/StepPlan/StepResult state machine and ADRs.
- [x] S00-04: Professional README, ROADMAP, docs, NOTICE and milestone/branch policy.
- [x] S00-05: Installable CPU-only package foundation, lint/type/test CI, deterministic synthetic smoke.
- [ ] S00-06: Execute HF pretrained Dense comparison on immutable weights and tokenizer.
- [ ] S00-07: Execute pinned vLLM runtime baseline under adequate compatible GPU.
- [ ] S00-08: Record CI run and review clean-room reproducibility.

## Acceptance / stop conditions

**Strong gate:** sources locked, design complete, local+CI tests green, baseline.json faithfully reports not_run when missing model/GPU, provenance recorded. **Not acceptable:** generated fake performance numbers, treating toy smoke as real autoregressive inference, beginning Stage 01 merely because scaffolding exists.

## Deliverables

`docs/sources.lock.json`, `docs/upstream-analysis.md`, `docs/rfcs/0001-engine-core.md`, `docs/adr/`, `docs/status.md`, `benchmarks/results/baseline-template.json`, `scripts/stage00_smoke.py`, `scripts/hf_baseline.py`, `tests/`, `.github/workflows/ci.yml`.

## Execution commands

```bash
python -m pip install -e ".[dev]"
python scripts/stage00_smoke.py --output benchmarks/results/stage00-smoke.json
pytest -q
ruff check .
mypy src scripts
```

**Next:** Do not start Stage 01 until missing hardware/model baseline gates are either satisfied or explicitly approved as a documented exception.

