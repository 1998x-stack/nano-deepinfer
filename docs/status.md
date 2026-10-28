# Verified repository status

**2026-10-28:** Stages 00–04 CPU reference implementations have been merged to `main`. Stage 04 adds optional, namespace-isolated shared physical KV prefix blocks with COW and LRU pin eviction. Stage 01 was merged as commit `959815ee1646eb77e8f98a8ca2a11bee37cc15cb`. This is an independently implemented, single-request GPT-2 and Qwen3 Dense reference; not a production inference server.

| Milestone | Verified status | Evidence |
| --- | --- | --- |
| Upstream research / RFC / Stage 00 CPU smoke | PASS | [source lock](sources.lock.json), [Architecture](architecture.md) |
| Stage 00 pretrained HF CPU cached/full reference | PASS | [CI 40468544667](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667) |
| Stage 01 GPT-2 independent logits/greedy-token parity | PASS, tiny pretrained checkpoint | [CI 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422) |
| Stage 01 Qwen3 independent full/cache/greedy parity | PASS, small random/debug weights | [CI 40476699366](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40476699366) (22 focused tests), [latest reference CI 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422) |
| Real Qwen3 debugging checkpoint | PASS | [model lock](qwen3-model.lock.json): `llamafactory/tiny-random-qwen3@81d6f5f5e05ed53ea8a1d19431266a486e46bbd8` |
| Qwen/Qwen3-0.6B trained checkpoint CPU/GPU/dtype matrix | **NOT RUN** | hardware/checkpoint gate remains open |
| vLLM matching-GPU comparative baseline | **NOT RUN / DEFERRED** | [ADR 0003](adr/0003-stage-00-gpu-deferral.md); required for Stage 05 claims |
| Stage 02 physical Paged KV, COW transaction, Block Pool and request table | **PASS (CPU, reference path)** | [21 tests + real HF checkpoint parity, CI 40484347235](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40484347235), [ADR 0004](adr/0004-paged-kv-transaction.md) |
| Stage 03 unified scheduler, mixed Prefill/Decode and logical request interleaving | **PASS (CPU, logical scheduling)** | [Stage 03](stages/stage-03.md), [ADR 0005](adr/0005-unified-scheduler.md), [real model CI 40489283022](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40489283022) |
| Stage 04 physically shared prefix KV refs / chunk cap | **PASS (CPU, pinned model correctness)** | [Stage 04](stages/stage-04.md), [ADR 0006](adr/0006-shared-prefix.md) |
| GPU fused model batches, PagedAttention, serving, MoE/EP | **NOT IMPLEMENTED** | Stages 05+ |

**Gate interpretation:** Stage 00 research/CPU paths complete but *original strict GPU gate remains PARTIAL*. Stage 01 small-model reference core passes; large trained Qwen3 validation and comprehensive cancellation/OOM paths are outstanding. Request to proceed to the next phase authorizes integrating tested research code, not falsely marking those gates complete. **Stage 02 CPU paging correctness passed; Stage 03 CPU multi-request scheduler passed its focused and real-model tests.** No vLLM speedup / GPU throughput or latency comparisons have been made.

See [Stage 01 plan and remaining requirements](stages/stage-01.md) and [acceptance](acceptance.md).

**Stage 02 boundary:** the physical page tensor store is real, but the synchronous model runner reconstructs dense KV for attention; no throughput/latency speedup or complete trained Qwen3-0.6B GPU performance is claimed. The actual Stage 02 HF run validates tiny GPT-2 pretrained and tiny Qwen3 random debugging checkpoints only.

**Stage 03 merged:** [PR #8](https://github.com/1998x-stack/nano-deepinfer/pull/8), commit `5dfa0ef4a89d2645b588dd4179333f40bfa9ac43`; verified [merge CI](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40489761672), [reference workflow](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40489557437). Stage 04 shared-prefix CPU correctness PASS (15 focused tests and fixed GPT-2/Qwen3 checkpoint oracle); full trained model and matched-GPU performance prerequisites remain open.

**Stage 04 merged:** [PR #10](https://github.com/1998x-stack/nano-deepinfer/pull/10), commit `d154d14b905fbd682745d517392b5d9204a5cb0c`. **Stage 04 evidence:** [real HF CPU workflow 40616756299](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40616756299) ran Stage 01: 22 PASS, Stage 02: 21 PASS, Stage 03: 17 PASS, Stage 04: 15 PASS, and saved `stage04-prefix-cache-cpu-evidence` with three actual CPU repeats for each checkpoint. The prior baseline is not GPU-matched. PR #10 final head [HF reference CI 40617005359](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40617005359) and [Python CI 40617005367](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40617005367) passed prior to merge.
