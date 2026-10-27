# Verified repository status

**2026-10-27:** Stage 01 **CPU reference core is merged into main** as commit `959815ee1646eb77e8f98a8ca2a11bee37cc15cb`. This is an independently implemented, single-request GPT-2 and Qwen3 Dense reference; not a production inference server.

| Milestone | Verified status | Evidence |
| --- | --- | --- |
| Upstream research / RFC / Stage 00 CPU smoke | PASS | [source lock](sources.lock.json), [Architecture](architecture.md) |
| Stage 00 pretrained HF CPU cached/full reference | PASS | [CI 40468544667](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667) |
| Stage 01 GPT-2 independent logits/greedy-token parity | PASS, tiny pretrained checkpoint | [CI 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422) |
| Stage 01 Qwen3 independent full/cache/greedy parity | PASS, small random/debug weights | [CI 40476699366](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40476699366) (22 focused tests), [latest reference CI 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422) |
| Real Qwen3 debugging checkpoint | PASS | [model lock](qwen3-model.lock.json): `llamafactory/tiny-random-qwen3@81d6f5f5e05ed53ea8a1d19431266a486e46bbd8` |
| Qwen/Qwen3-0.6B trained checkpoint CPU/GPU/dtype matrix | **NOT RUN** | hardware/checkpoint gate remains open |
| vLLM matching-GPU comparative baseline | **NOT RUN / DEFERRED** | [ADR 0003](adr/0003-stage-00-gpu-deferral.md); required for Stage 05 claims |
| Paged KV, continuous batching, online API, MoE/EP | **NOT IMPLEMENTED** | future Stage 02+ |

**Gate interpretation:** Stage 00 research/CPU paths complete but *original strict GPU gate remains PARTIAL*. Stage 01 small-model reference core passes; large trained Qwen3 validation and comprehensive cancellation/OOM paths are outstanding. Request to proceed to the next phase authorizes integrating tested research code, not falsely marking those gates complete. **Stage 02 is not started.** No vLLM speedup / GPU throughput or latency comparisons have been made.

See [Stage 01 plan and remaining requirements](stages/stage-01.md) and [acceptance](acceptance.md).
