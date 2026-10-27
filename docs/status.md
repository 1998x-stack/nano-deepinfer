# Development status and evidence

**Date:** 2026-10-27 · **Current implementation milestone:** Stage 00 (foundation, research, smoke harness).

| Requirement | Evidence | Result |
| --- | --- | --- |
| GitHub upstream SHA freeze | `sources.lock.json` | documented |
| Reference architectures / source mapping | `upstream-analysis.md`; `rfcs/0001-engine-core.md` | documented |
| Contract design and invariants | `architecture.md`, ADRs; pytest contracts | coded/check via CI |
| CPU synthetic smoke and env JSON | `scripts/stage00_smoke.py` | run to generate evidence |
| Full Transformers comparison | `scripts/hf_baseline.py` available as optional harness | **not run / no checkpoint** |
| vLLM + DeepInfer runtime comparisons | hard GPU/model prerequisites | **not run** |
| GPU memory / tokens per second | no physical GPU benchmark | **no measured results** |

**Stage 00 exit status:** PARTIAL until required reference and CI evidence exists. Creating a branch or passing toy tests does not close the hardware-dependent gates. **Stage 01 code not implemented**.

Record test outputs and workflow run URL in this document when verified. Run `pytest -q`, `ruff check .`, `mypy src scripts`, then Stage 00 smoke.

