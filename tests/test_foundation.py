"""Pure-CPU Stage 00 contract tests; no claims about model inference."""

from dataclasses import FrozenInstanceError

import pytest

from nano_deepinfer.foundation import (
    Request,
    SamplingParams,
    StepPlan,
    StepResult,
    synthetic_reference_decode,
    validate_step_result,
)


def test_synthetic_oracle_is_repeatable() -> None:
    assert synthetic_reference_decode((1, 2, 3), 6) == (1, 8, 9, 14, 5, 11)
    assert synthetic_reference_decode((1, 2, 3), 6) == synthetic_reference_decode((1, 2, 3), 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 1, "temperature": -1.0},
        {"max_new_tokens": 1, "top_p": 0.0},
        {"max_new_tokens": 1, "top_p": 1.1},
        {"max_new_tokens": 1, "top_k": 0},
        {"max_new_tokens": 1, "seed": -1},
    ],
)
def test_bad_sampling_rejected(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_request_and_plan_contracts() -> None:
    params = SamplingParams(max_new_tokens=4)
    request = Request("r", (1, 2), params, "sha256:model")
    assert request.prompt_token_ids == (1, 2)
    plan = StepPlan(0, (("r", 1),), token_budget=8)
    validate_step_result(plan, StepResult(0, (("r", 1),), True))
    with pytest.raises(FrozenInstanceError):
        plan.step_id = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        StepPlan(0, (("r", 1), ("r", 2)), 8)


@pytest.mark.parametrize(
    "result",
    [
        StepResult(1, (("r", 1),), True),
        StepResult(0, (("r", 0),), True),
        StepResult(0, (("r", 1),), False),
    ],
)
def test_reject_stale_or_unacknowledged_result(result: StepResult) -> None:
    with pytest.raises(ValueError):
        validate_step_result(StepPlan(0, (("r", 1),), 8), result)


@pytest.mark.parametrize("prompt,steps", [((), 1), ((18,), 1), ((1,), -1)])
def test_invalid_synthetic_inputs(prompt: tuple[int, ...], steps: int) -> None:
    with pytest.raises(ValueError):
        synthetic_reference_decode(prompt, steps)
