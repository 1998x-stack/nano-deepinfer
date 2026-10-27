"""Stage 00 provenance and release-claim invariants, with no network calls."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_lock_has_immutable_full_shas() -> None:
    lock = json.loads((ROOT / "docs/sources.lock.json").read_text(encoding="utf-8"))
    assert lock["schema_version"] == 1
    upstreams = lock["upstreams"]
    assert {s["key"] for s in upstreams} == {"nano-vllm", "vllm-v1", "deepinfer"}
    for upstream in upstreams:
        assert re.fullmatch(r"[0-9a-f]{40}", upstream["commit"])
        assert upstream["license"] in {"MIT", "Apache-2.0"}
        assert upstream["paths"]


def test_every_planned_stage_has_a_document() -> None:
    for number in range(13):
        stage = ROOT / f"docs/stages/stage-{number:02d}.md"
        assert stage.is_file(), f"missing {stage}"
        assert f"Stage {number:02d}" in stage.read_text(encoding="utf-8")


def test_unmeasured_baseline_is_explicit() -> None:
    baseline = json.loads(
        (ROOT / "benchmarks/results/baseline-template.json").read_text(encoding="utf-8")
    )
    assert baseline["status"] == "not_run"
    assert all(v is None for v in baseline["metrics"].values())
    assert baseline["correctness"]["hf_token_exact"] is None


def test_readme_never_claims_inference_is_ready() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "not a production inference server" in readme
    assert "Stage 00" in readme


def test_license_and_notice_present() -> None:
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    for name in ("nano-vLLM", "vLLM", "DeepInfer"):
        assert name in notice
