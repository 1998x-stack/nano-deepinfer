"""CPU-only contracts and synthetic correctness fixture for Stage 00.

This is NOT an LLM inference implementation and must not be benchmarked as one.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 <= self.temperature < float("inf"):
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    model_fingerprint: str

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.model_fingerprint.strip():
            raise ValueError("request id and model fingerprint are required")
        if not self.prompt_token_ids or any(t < 0 for t in self.prompt_token_ids):
            raise ValueError("prompt tokens must be nonempty and nonnegative")


@dataclass(frozen=True, slots=True)
class StepPlan:
    step_id: int
    request_versions: tuple[tuple[str, int], ...]
    token_budget: int

    def __post_init__(self) -> None:
        if self.step_id < 0 or self.token_budget < 1:
            raise ValueError("invalid step ID or budget")
        if not self.request_versions or len({key for key, _ in self.request_versions}) != len(
            self.request_versions
        ):
            raise ValueError("step requires unique requests")
        if any(not key or version < 0 for key, version in self.request_versions):
            raise ValueError("invalid request version")


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: int
    request_versions: tuple[tuple[str, int], ...]
    worker_ack: bool


def validate_step_result(plan: StepPlan, result: StepResult) -> None:
    """Validate a candidate result; not a stateful production committer."""
    if result.step_id != plan.step_id:
        raise ValueError("stale or mismatched step ID")
    if result.request_versions != plan.request_versions:
        raise ValueError("stale or reordered request versions")
    if not result.worker_ack:
        raise ValueError("worker has not acknowledged completion")


def synthetic_reference_decode(
    prompt: tuple[int, ...], steps: int, vocab_size: int = 17
) -> tuple[int, ...]:
    """Deterministic test fixture only; not a neural network or a sampler."""
    if not prompt or any(token < 0 or token >= vocab_size for token in prompt):
        raise ValueError("prompt must contain valid tokens")
    if steps < 0 or vocab_size < 2:
        raise ValueError("invalid number of steps or vocabulary")
    state = prompt[-1]
    output: list[int] = []
    for _ in range(steps):
        state = (state * 5 + 3) % vocab_size
        output.append(state)
    return tuple(output)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, str | None]:
    """Portable, non-sensitive toolchain identifiers. No user paths or tokens."""
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        sha = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        sha = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "git_sha": os.environ.get("GITHUB_SHA") or sha,
    }
