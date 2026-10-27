# nano-deepinfer

> **A small, independently implemented LLM inference engine for understanding the systems behind nano-vLLM, vLLM V1 and DeepInfer.**

[![CI](https://github.com/1998x-stack/nano-deepinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/nano-deepinfer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
![Stage](https://img.shields.io/badge/Stage-00%20Foundation-orange)

**Project status: Stage 00 — architecture, research, reproducibility and project foundation.** This repository is **not yet a working model inference server**. No performance claims are made.

**Why?** nano-vLLM makes core inference concepts understandable; vLLM V1 shows robust production-oriented abstractions; DeepInfer studies highly optimized model-specific execution at scale. nano-deepinfer builds an independent, testable learning and research implementation inspired by their publicly documented designs, without treating vLLM or DeepInfer as its runtime.

## Architecture at a glance

```text
[Offline API / future Serving API]
                 |
          Request validation
                 |
          Request lifecycle
                 |
        Unified Scheduler  <---->  KV / State manager
                 |
           Immutable StepPlan
                 |
           Model Runner
            /         \
    Attention backend   Model / MoE backend
            \         /
             Sampler
                 |
      Validated StepResult
                 |
      Commit / output / metrics
```

Stage 00 defines these interfaces; Stage 01 implements single-request dense prefill/decode; stages 02–04 add memory management and continuous batching.

## Roadmap

| Version | Stages | Planned scope | Status |
| --- | --- | --- | --- |
| `v0.1` | 00–04 | Dense offline engine, paged KV, batching, prefix reuse | Stage 00 underway |
| `v0.2` | 05–06 | GPU kernels, CUDA Graph, asynchronous steps | Planned |
| `v0.3` | 07–09 | Toy MoE, expert parallelism, speculative decoding, SWA experiments | Planned |
| `v1.0` | 10–12 | Serving, reproducible evaluations, optional P/D experiment | Planned |

Detailed plan: [ROADMAP](ROADMAP.md) · [Architecture](docs/architecture.md) · [Current status](docs/status.md) · [Acceptance](docs/acceptance.md) · [Source lock](docs/sources.lock.json).

## Start here — Stage 00

```bash
git clone https://github.com/1998x-stack/nano-deepinfer.git
cd nano-deepinfer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/stage00_smoke.py --output benchmarks/results/stage00-smoke.json
pytest -q
ruff check .
mypy src scripts
```

The Stage 00 smoke test checks deterministic token-counting and creates an environment snapshot. **It does not execute an LLM.** PyTorch, Transformers, CUDA and model checkpoints are intentionally not required for the foundation smoke test.

## Design rules

- Implement independent scheduler, model runner and KV manager code; do not wrap vLLM as the execution engine.
- Require a reference correctness oracle before enabling any optimization.
- Keep CPU-testable scheduler and allocation policies separate from GPU-specific kernels.
- Define request IDs, immutable step plans, versioned outputs and idempotent commits before implementing asynchronous execution.
- Report both improvements and regressions with fully specified workloads; never claim performance without measurements.
- Respect upstream licensing and attribution; third-party source copying requires a documented license review.

## Documentation

[Stage 00 tasks](docs/stages/stage-00.md) · [Stage 01 entry criteria](docs/stages/stage-01.md) · [Benchmark methodology](docs/benchmarks.md) · [Upstream code study](docs/upstream-analysis.md) · [Engineering RFC](docs/rfcs/0001-engine-core.md) · [Branch policy](docs/branching.md)

## Development

Contributions should use short `feat/stage-XX-*` branches, tests, an ADR for contract changes and a reviewable PR. `main` tracks the newest **verified** code. The named `v0.1`, `v0.2`, `v0.3`, and `v1.0` branches are **planning markers**, not releases; a version is released only after its exit gates pass.

## Credits and license

Studied: [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) (MIT), [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) and [DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0). This project is independently authored; see [NOTICE](NOTICE.md), [sources](docs/references.md), and [LICENSE](LICENSE). No warranty of performance or full DeepInfer compatibility.
