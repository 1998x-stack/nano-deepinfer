# Development status and verified evidence

**Date:** 2026-10-27 · **Current milestone:** Stage 00 foundation merged to `main`. **Strict Stage 00 exit:** PARTIAL because actual HF and vLLM reference baselines have not been executed.

| Requirement | Evidence | Result |
| --- | --- | --- |
| GitHub upstream SHA freeze | [source lock](sources.lock.json) | Completed |
| Source-level analysis and provenance | [upstream analysis](upstream-analysis.md), [references](references.md) | Completed |
| Architecture and typed interface contract | [RFC 0001](rfcs/0001-engine-core.md), [ADRs](adr) | Completed |
| CPU-only synthetic contracts and smoke | `tests/test_foundation.py`, `scripts/stage00_smoke.py` | 14/14 local tests PASS on Linux x86_64 / Python 3.13.5; smoke PASS |
| Branch push CI | [CI run 40467018446](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40467018446) | PASS |
| PR #1 CI | [CI run 40467077724](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40467077724) | PASS |
| Merged main CI | [CI run 40467149468](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40467149468) | PASS |
| Milestone pointers v0.1/v0.2/v0.3/v1.0 | All initially pointed to `cce9567bd0092601d1601c5318f6fbea0dd27645` | Created; **NOT releases** |
| Real HF pretrained Dense baseline | Optional `scripts/hf_baseline.py` awaits checkpoint with immutable model SHA | **NOT RUN** |
| Pinned vLLM GPU baseline | No compatible GPU execution environment connected in this run | **NOT RUN** |
| LLM throughput/TTFT/TPOT | No actual model inference here | **NO MEASURED RESULTS** |

### Evidence semantics

- The **synthetic** CPU oracle validates basic determinism and contracts, **not** autoregressive neural-network inference or model agreement.
- This project has **not** implemented `LLM.generate` or a dense model runner. Those belong to Stage 01.
- Do not mark Stage 00 exit PASS until model/HF/vLLM benchmarks are genuinely measured or a recorded exception explicitly narrows the gate.
- CI links correspond to commits and runs inspected on the repository. Subsequent test changes must generate fresh CI evidence.

### Next permitted action

Close the remaining Stage 00 measurement gaps on a machine with Transformers, pinned model weights and compatible vLLM/GPU. Only then begin Stage 01 development as separately reviewed work.
