# Acceptance and release quality gates

## Stage 00: design complete, strict exit PARTIAL
- [x] Upstream commits locked, architecture and RFC, source research, CPU synthetic tests and CI.
- [x] Pinned genuine pretrained HF tiny GPT-2 CPU reference measured: [run 40468544667](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667).
- [ ] Matched **vLLM GPU** reference performance benchmark: **NOT RUN**. [ADR 0003](adr/0003-stage-00-gpu-deferral.md) defers this required validation to the Stage 05 optimization prerequisite.

## Stage 01: core PASS, complete hardware/model exit OPEN
- [x] Independent GPT-2 and Qwen3 Dense model forward including separate prefill/cache decode; only batch size one, eager/unpaged KV.
- [x] Greedy, seeded temperature, top-k, top-p, eos and stop token IDs.
- [x] Tiny pretrained GPT-2 HF logits/generated IDs match; small Qwen3 HF and immutable random-weight checkpoint full/cache/logit/greedy parity: [run 40477183422](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40477183422), [model tests 22 PASS](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40476699366).
- [x] Reference CI, example, reproducible source/model locks and honest README status.
- [ ] Complete pretrained **Qwen/Qwen3-0.6B** tokenizer/weight comparison on target hardware.
- [ ] Robust cancellation and device OOM recovery under failure injection.

The validated Stage 01 reference code is merged into `main`. The above unverified criteria are **not** considered PASS. Stage 02 has **not** started.

## Common DoD
Tests, correctness oracle, SHA-pinned weights and workload, Ruff/MyPy/Pytest, explicit failures and source notices. Never claim vLLM performance without same hardware/loads and reproducible results.

## Release gates
`v0.1`: Stage 00–04 dense/paged-KV/batching/prefix correctness.
`v0.2`: Stage 05–06 GPU optimized/async reproducibility.
`v0.3`: Stage 07–09 MoE/speculation/SWA experimental correctness.
`v1.0`: Stage 10/12 stable serving/release; Stage 11 optional research.
