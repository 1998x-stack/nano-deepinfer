# Stage 00 — Source research and reproducible foundation

**Implementation delivered on `main`; strict stage exit = PARTIAL.**

## Verified work

- [x] S00-01 Pin upstream source commit SHAs and licenses.
- [x] S00-02 Analyze nano-vLLM, vLLM V1 and DeepInfer execution/source boundaries.
- [x] S00-03 Write RFC, typed contracts, cache/state architecture and ADRs.
- [x] S00-04 Document 13 stage deliverables, exit criteria, branch/release policy and README.
- [x] S00-05 Installable CPU foundation, deterministic synthetic smoke, CI matrix.
- [x] S00-06 **Run a real pinned HF small Dense checkpoint on CPU**, check full-prefill versus KV-cache incremental logits and publish evidence. [Successful run](https://github.com/1998x-stack/nano-deepinfer/actions/runs/40468544667).
- [ ] S00-07 Compare pinned vLLM inference on an actual compatible GPU and publish raw benchmark evidence.
- [ ] S00-08 Obtain explicit owner closeout or documented waiver for GPU prerequisite.

## Preconditions for Stage 01 acceptance

1. Complete S00-07 or record owner-approved constrained exception; leave Stage 00 `PARTIAL` until then.
2. Stage 01 must implement independently authored model forward, not use HF/vLLM for its runtime execution.
3. Compare both full-prefill and incremental logits, greedy generated Token IDs, and describe tolerance and model SHA.
4. Do not claim any vLLM performance improvement without controlled equal-hardware measurements.

## Commands

```bash
python -m pip install -e ".[dev]"
python scripts/stage00_smoke.py --output benchmarks/results/stage00-smoke.json
pytest -q
ruff check .
mypy src scripts
```

The separate CPU reference workflow installs PyTorch and Transformers and writes a real-model baseline artifact. This is **not** an independent nano-deepinfer implementation.
