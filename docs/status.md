# Development status and verified evidence

**Date:** 2026-10-27 · **Current merged milestone:** Stage 00 foundation. **Strict Stage 00 exit:** PARTIAL; real vLLM GPU comparison remains pending.

| Requirement | Evidence | Result |
| --- | --- | --- |
| Three upstream SHAs pinned | [source lock](sources.lock.json) | Done |
| Source study, architecture, contracts and decisions | [upstream analysis](upstream-analysis.md), [RFC](rfcs/0001-engine-core.md) | Done |
| CPU synthetic reference and provenance checks | `tests/test_foundation.py`, `tests/test_stage00_artifacts.py` | PASS in CI |
| Real pretrained HF checkpoint on CPU | [Pinned tiny-GPT-2 model](hf-model.lock.json), [workflow run 40468544667](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667) | **PASS**: Transformers full-prefix vs incremental KV logits within 1e-4 |
| Raw real-reference JSON | [Workflow artifacts](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667) → `stage00-pinned-hf-pretrained-baseline` | Published |
| Base CI after merge | [Stage 00 main CI](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468688608) | PASS |
| Real vLLM GPU baseline vs same checkpoint | compatible GPU runtime not provided | **NOT RUN** |
| nano-deepinfer versus Transformers actual model | Stage 01 research PR, not part of Stage 00 merged code | **NOT IN MAIN** |
| GPU throughput / TTFT / TPOT measurements | no GPU benchmark | **NOT RUN; no speedup claim** |

**Important distinction:** Stage 00 HF run validates **Transformers against itself** with an immutable pretrained checkpoint and full-vs-cached computation; it does not prove independent nano-deepinfer correctness. The Stage 01 branch may contain its own tests but remains unmerged pending Stage 00 closeout policy.

**Next strict gate:** execute GPU vLLM reference benchmark or obtain documented owner-approved scope exception. Retain `PARTIAL` until then. See [acceptance](acceptance.md).
