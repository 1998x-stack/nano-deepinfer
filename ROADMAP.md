# nano-deepinfer — Research-to-Engine Roadmap

**Baseline date:** 2026-10-27. **Scope today:** Stage 00 only. Stages are exit-gated; future branch names do not signify implemented functionality.

| Version | Stages | Deliverables | Gate |
| --- | --- | --- | --- |
| v0.1 | 00–04 | Foundation; CPU/Dense prefill/decode; block-managed KV; unified token-budget batching; chunked prefill and prefix cache | Dense correctness against pinned HF checkpoint; deterministic scheduler tests |
| v0.2 | 05–06 | Configurable attention/kernel backends; CUDA Graph; async plan/commit pipeline | Correct fallback and measured regression/speedup results |
| v0.3 | 07–09 | Toy MoE and EP/EPLB; spec decode; model-specific bounded SWA replay | Numerical/reference correctness, rank conservation, rollback and cache invariants |
| v1.0 | 10–12 | Single-node HTTP streaming and structured output; optional P/D experiment; release evaluation | Reproducible CI, quality, performance report and documentation |

## Stage execution sequence

| Stage | Area | Milestone | Prerequisites |
| --- | --- | --- | --- |
| 00 | Source research, architecture RFC, source lock, CI, reproducible synthetic smoke | Foundation | None |
| 01 | CPU oracle, dense PyTorch/HF adapter, single-request prefill/decode | Dense correctness | Stage 00 green, model artifact access |
| 02 | Paged KV block ownership, atomic reservation and release | KV memory | Stage 01 |
| 03 | Unified continuous batching, preemption and scheduler replay | Scheduler | Stage 02 |
| 04 | Chunked prefill and isolation-safe prefix cache | v0.1 | Stage 03 |
| 05 | Attention backend optimization, CUDA Graph and kernel microbench | Performance | v0.1 + NVIDIA hardware |
| 06 | Async CPU/GPU overlap; sequence/version-based result commit | v0.2 | Stage 05 |
| 07 | MoE reference router; optional multi-GPU expert parallelism/EPLB | MoE | Stage 01, 2+ GPUs for EP |
| 08 | Target/draft speculation, verification/rollback; DSpark-inspired adaptive budget | Speculation | Stage 06, compatible draft |
| 09 | Sliding-window state semantics and bounded replay | SWA research | Toy SWA oracle and compatible model |
| 10 | Single-node API, SSE streaming, structured output | Serving | v0.2 core |
| 11 | Experimental prefill/decode disaggregation and KV handoff | Distributed research | >1 device/process and network profiling |
| 12 | Release hardening, reproducibility bundle, reliability and baseline comparisons | v1.0 | Stage 10, all required gates |

## Execution and branch conventions

- `main` is newest **merged and validated** work, never an unreleased tag.
- `feat/stage-XX-topic` are short-lived implementation branches and PRs.
- Long-lived `v0.1`, `v0.2`, `v0.3`, `v1.0` are initial *planning branch pointers* and shall be advanced only for work belonging to that stream, rebased or merged regularly; they are **not version tags**.
- Never force-push `main`. Never merge failing gates. Increment functionality in visible README only after verification.
- No unmeasured claims, no implicit copying of upstream implementations. See [branch policy](docs/branching.md).

## Reference documents

[Architecture](docs/architecture.md) · [RFC](docs/rfcs/0001-engine-core.md) · [Acceptance](docs/acceptance.md) · [Benchmarks](docs/benchmarks.md) · [Stage 00](docs/stages/stage-00.md) · [Stage 01](docs/stages/stage-01.md)

