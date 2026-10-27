# Baseline and benchmark methodology

**Warning:** No measured LLM inference performance exists at Stage 00. Synthetic smoke must not be compared as a throughput benchmark with real engines.

## Workload suite (future)

| ID | Input | Output | Concurrent | Workload |
| --- | ---: | ---: | ---: | --- |
| W1 | 128 | 128 | 1 | latency baseline |
| W2 | 128 | 128 | 16,64 | decode throughput |
| W3 | 4096 | 128 | 8,32 | long prefill |
| W4 | 8192 | 32 | 16 | mixed queue |
| W5 | 1024 | 256 | 16 | 80% shared prefix |
| W6 | model maximum | 64 | 1,4 | context bound |
| W7 | 512 | 128 | 16 | MoE expert skew |

Do not silently exclude unsupported/OOM cases; report `unsupported` or `failed` with reason and configuration.

## Metrics and statistical practice

TTFT, TPOT/ITL p50/p95/p99 (only enough requests justify percentile resolution), total input/output tokens per second, requests/s, GPU idle time, CPU/GPU RAM, startup/teardown, cache hit, expert skew, draft acceptance. Record concurrent workload's offered load and completed work separately.

Run warmup then at least five timed repetitions; include median, dispersion and raw traces. Keep model revision, dtype, attention backend, tokenizer, context window, temperature/seed, GPU count and clocks as comparable as practical. Greedy token equivalence and stochastic acceptance statistics are different test categories.

## Artifacts

`benchmarks/results/stage00-smoke.json` — deterministic synthetic smoke only (generated locally, not a performance claim). `benchmarks/results/baseline-template.json` — explicit status not_run. Future naming: `<stage>-<engine>-<modelsha>-<workload>-<utc>.json`; commit portable small summaries only.

## Reproducible baseline procedure

1. Record `git rev-parse HEAD`, `python -V`, package versions and available accelerator inventory.
2. Record model and tokenizer **immutable revision SHA**, context, prompt tokens, outputs, dtype and sampling config.
3. For HF, run `scripts/hf_baseline.py --model <checkpoint> --revision <sha> --prompt "..." --output <path>`; no remote code trust.
4. For vLLM, use separately installed upstream pinned commit in appropriate GPU environment, record launch arguments and request traces.
5. Test confidence intervals and correctness before drawing performance conclusions.

See [Environment/limitations](status.md).

