# ADR 0003 — Defer unavailable GPU comparison without overstating Stage 00/01 acceptance

**Decision date:** 2026-10-27. **Status:** Scoped continuation, open verification debt.

Stage 00 originally demanded a pinned vLLM GPU reference run before later-stage development. Real GPU execution was unavailable in the connected validation environment. An actual pinned Transformers pretrained CPU check was completed successfully instead, and independent Stage 01 model correctness checks have also run.

User-requested continuation to the next phase is interpreted as authorization to work on and integrate a CPU-reference engine, **not authorization to claim full Stage 00 acceptance or any vLLM GPU result**.

A vLLM matched-hardware baseline is retained as a **required Stage 05 optimization prerequisite**; Qwen3-0.6B full-scale checkpoint testing remains open in Stage 01 hardware matrix.

Safeguards: README marks code as CPU-first reference, feature branches use CI, no performance assertions, Stage 02 must remain off until explicitly scheduled. Stage 00 remains partial; user can later require original strict gate before expansion.
