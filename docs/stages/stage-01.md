# Stage 01 — Single-request Dense reference inference

**Status: reference core verified on CPU; full hardware/model gate remains OPEN.** No Stage 02 code is included here.

## Actual implemented modules

- [x] Immutable Request/SamplingParams and SequenceState; generation result with IDs and finish reason.
- [x] Independent GPT-2-style causal Dense model, learned position embeddings, explicit per-layer KV.
- [x] Independent Qwen3 Dense model with Q/K RMSNorm, RoPE, grouped-query attention, SwiGLU, optional biases and per-layer compressed KV.
- [x] Single-request engine handles greedy sampling, temperature, top-k/top-p, seed, eos and stop-token IDs.
- [x] Complete prefill vs incrementally cached decode token/logit comparisons.
- [x] HF-pinned genuine pretrained tiny GPT-2 weights, greedy-generated token equivalence.
- [x] HF Qwen3 random-config seeded-weight parameterized tests, and immutable real tiny Qwen3 debug-weight checkpoint.
- [x] CPU smoke tests plus dedicated Torch/Transformers CI and JSON evidence.

## Verified evidence

- [CI job: Stage 01 Qwen3/GPT-2 CPU tests](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40476699366): 22 focused tests PASS, HF tiny GPT-2 greedy-token benchmark PASS, HF tiny-random Qwen3 checkpoint PASS.
- Qwen3 debug weights: `llamafactory/tiny-random-qwen3` at `81d6f5f5e05ed53ea8a1d19431266a486e46bbd8`. This is a real fetched weight file, but **random**, not a meaningful trained chat model.
- Base Stage 00 genuine Transformers checkpoint: `sshleifer/tiny-gpt2@5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`.
- `benchmarks/results/` contains **real GitHub Actions artifacts**, not invented performance numbers.

## Known limitations / unfinished gates

- [ ] Validate Qwen/Qwen3-0.6B full nonrandom pretrained checkpoint, tokenizer and context on compatible hardware. This is not inferred from the tiny checkpoint.
- [ ] Establish actual matched-hardware vLLM reference benchmark originally required by Stage 00; scoped deferral to Stage 05 research, **not marked as executed**.
- [ ] Cancellation and device OOM recovery beyond synchronous in-process failure behavior; richer request lifecycle.
- [ ] Multi-sequence scheduling and paged KV allocations are **Stage 02/03**, not available in Stage 01.
- [ ] No streaming API, distributed service, async dispatch, or CUDA Graph.
- [ ] Numerical tolerance only validated on pinned CPU float32 oracle; mixed precision remains separate.

## Scope-control decision

The user's latest request to continue to the next stage authorizes proceeding with implementation despite an unavailable GPU comparison; it is **not a claim that the original GPU gate was met**. Treat Stage 00 as `PARTIAL / GPU BENCHMARK DEFERRED`, Stage 01 as `REFERENCE CORE PASS / FULL MODEL GATE OPEN`, and keep feature work isolated until unit and reference CI passes. See [ADR 0003](../adr/0003-stage-00-gpu-deferral.md).

## Commands

```bash
python -m pip install -e ".[reference,dev]"
python examples/stage01_reference.py
pytest -q tests/test_dense_reference.py tests/test_hf_gpt2_parity.py tests/test_qwen3_reference.py
python scripts/stage01_qwen3_checkpoint.py --output benchmarks/results/stage01-qwen3.json
```

No deployment claims or speedup claims are accepted from one-shot CPU timings.
