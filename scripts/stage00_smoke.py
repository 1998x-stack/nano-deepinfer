"""Run a deterministic CPU-only *synthetic* smoke test and snapshot toolchain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_deepinfer.foundation import environment_snapshot, synthetic_reference_decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/stage00-smoke.json"))
    args = parser.parse_args()
    expected = (1, 8, 9, 14, 5, 11)
    observed = synthetic_reference_decode((1, 2, 3), steps=6)
    if observed != expected:
        raise RuntimeError("synthetic oracle mismatch")
    record = {
        "schema_version": 1,
        "stage": "00",
        "kind": "synthetic_contract_smoke_not_llm_inference",
        "status": "passed",
        "prompt_token_ids": [1, 2, 3],
        "observed": list(observed),
        "expected": list(expected),
        "environment": environment_snapshot(),
        "llm_benchmark_status": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Stage 00 synthetic smoke PASS: {args.output}")


if __name__ == "__main__":
    main()
