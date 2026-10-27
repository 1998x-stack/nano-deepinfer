# nano-deepinfer

> **A small, independently implemented LLM inference engine for understanding the systems behind nano-vLLM, vLLM V1 and DeepInfer.**

[![CI](https://github.com/1998x-stack/nano-deepinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/nano-deepinfer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
![Stage](https://img.shields.io/badge/Stage-02%20Paged%20KV%20Research-blue)

**Project status: Stage 02 — a CPU-first GPT-2/Qwen3 Dense reference engine with optional transactional *paged physical KV storage*.** It is **not a production inference server**; the attention kernel still materializes dense KV, and there is no continuous batching, GPU PagedAttention, optimized kernel or distributed execution. No vLLM performance claims are made.

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

Stage 00 defines the interfaces; Stage 01 implements single-request Dense prefill/decode; Stage 02 adds optional real physical KV pages; Stages 03–04 plan batching and prefix reuse.

## Roadmap

| Version | Stages | Planned scope | Status |
| --- | --- | --- | --- |
| `v0.1` | 00–04 | Dense offline engine, paged KV, batching, prefix reuse | Stage 01+02 CPU correctness verified; Stage 03–04 pending |
| `v0.2` | 05–06 | GPU kernels, CUDA Graph, asynchronous steps | Planned |
| `v0.3` | 07–09 | Toy MoE, expert parallelism, speculative decoding, SWA experiments | Planned |
| `v1.0` | 10–12 | Serving, reproducible evaluations, optional P/D experiment | Planned |

Detailed plan: [ROADMAP](ROADMAP.md) · [Architecture](docs/architecture.md) · [Current status](docs/status.md) · [Acceptance](docs/acceptance.md) · [Source lock](docs/sources.lock.json).

## Try Stage 01: executable Dense reference model

The lightweight eager runner supports GPT-2 and Qwen3 **weights**, prefill, incremental unpaged per-layer KV, greedy and seeded sampling. It does not use vLLM or Hugging Face to execute forward passes; Hugging Face is used only to load weights and independently verify results.

```bash
python -m pip install -e ".[reference,dev]"
python examples/stage01_reference.py
pytest -q tests/test_dense_reference.py tests/test_hf_gpt2_parity.py tests/test_qwen3_reference.py
python scripts/stage01_qwen3_checkpoint.py --output benchmarks/results/stage01-qwen3.json
```

Stage 01 has two reference models, both with **batch size one**, no streaming/serving, no speedup claim. The CI compares a genuine public tiny GPT-2 pretrained checkpoint and small *randomly initialized* Qwen3 checkpoint to Transformers logits and greedy tokens. Qwen3-0.6B large-checkpoint/accelerator performance tests remain outstanding.

## Try Stage 02: transactional paged KV storage

Stage 02 separates logical token positions from physical fixed-size page IDs. The CPU BlockManager supports request ownership, generation-stamped handles, atomic reserve/commit/abort, a copy-on-write (COW) partial tail, and complete cleanup after cancellation or OOM. The Torch PagedKVStore writes real per-layer K/V pages and reconstructs dense tensors for the existing independent model runner.

```bash
python -m pip install -e ".[reference,dev]"
pytest -q tests/test_stage02_block_manager.py tests/test_stage02_paged_kv.py
python examples/stage02_paged_reference.py
python scripts/stage02_real_model_parity.py
```

**Correctness, not performance:** the Stage 01 attention kernel is unchanged; materializing full dense KV each step can be slower. The COW append needs a spare page even when partially occupied. No PagedAttention, prefix sharing or continuous batching has been implemented. See [Stage 02 design/acceptance](docs/stages/stage-02.md) and [ADR 0004](docs/adr/0004-paged-kv-transaction.md). The GPU vLLM comparison and full Qwen3-0.6B trained-checkpoint validation remain outstanding.

## Stage 00 foundation

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

The Stage 00 smoke test checks deterministic toy-token processing and creates an environment snapshot. **That smoke test does not execute an LLM.** The separate Stage 01 example above actually executes the independently implemented neural network. PyTorch, Transformers, CUDA and model checkpoints are intentionally not required for the foundation smoke test.

## Design rules

- Implement independent scheduler, model runner and KV manager code; do not wrap vLLM as the execution engine.
- Require a reference correctness oracle before enabling any optimization.
- Keep CPU-testable scheduler and allocation policies separate from GPU-specific kernels.
- Define request IDs, immutable step plans, versioned outputs and idempotent commits before implementing asynchronous execution.
- Report both improvements and regressions with fully specified workloads; never claim performance without measurements.
- Respect upstream licensing and attribution; third-party source copying requires a documented license review.

## Documentation

[Stage 00 tasks](docs/stages/stage-00.md) · [Stage 01](docs/stages/stage-01.md) · [Stage 02](docs/stages/stage-02.md) · [Benchmark methodology](docs/benchmarks.md) · [Upstream code study](docs/upstream-analysis.md) · [Engineering RFC](docs/rfcs/0001-engine-core.md) · [Branch policy](docs/branching.md)

## Development

Contributions should use short `feat/stage-XX-*` branches, tests, an ADR for contract changes and a reviewable PR. `main` tracks the newest **verified** code. The named `v0.1`, `v0.2`, `v0.3`, and `v1.0` branches are **planning markers**, not releases; a version is released only after its exit gates pass.

## Credits and license

Studied: [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) (MIT), [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) and [DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0). This project is independently authored; see [NOTICE](NOTICE.md), [sources](docs/references.md), and [LICENSE](LICENSE). No warranty of performance or full DeepInfer compatibility.
