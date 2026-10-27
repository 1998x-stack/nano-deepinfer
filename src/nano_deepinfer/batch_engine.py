# mypy: ignore-errors
"""Stage 03 synchronous multi-request executor using a shared Stage 02 KV pool.

Continuous scheduling means newly submitted requests can run while older
requests decode. Each model call remains a batch-of-one eager forward;
no parallel GPU batch is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from nano_deepinfer.cache.block_manager import KVCapacityError
from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.dense_gpt2 import ReferenceGPT2
from nano_deepinfer.dense_qwen3 import ReferenceQwen3
from nano_deepinfer.engine import GenerationResult, sample_next
from nano_deepinfer.foundation import Request, SamplingParams
from nano_deepinfer.scheduler import (
    RequestView,
    ScheduledBatch,
    ScheduledWork,
    UnifiedScheduler,
)

Status = Literal["waiting", "running", "finished", "cancelled", "failed"]


@dataclass(slots=True)
class _Sequence:
    request: Request
    rng: torch.Generator
    stop_ids: tuple[int, ...]
    generated: list[int] = field(default_factory=list)
    computed: int = 0
    last_logits: Tensor | None = None
    version: int = 0
    preemptions: int = 0
    status: Status = "waiting"
    error: str | None = None

    @property
    def context(self) -> tuple[int, ...]:
        return self.request.prompt_token_ids + tuple(self.generated)


@dataclass(frozen=True, slots=True)
class StepEvent:
    request_id: str
    kind: Literal["prefill", "decode", "skip", "finish", "preempt"]
    cost: int
    token_id: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EngineStep:
    step_id: int
    token_budget: int
    scheduled_tokens: int
    scheduled_requests: tuple[str, ...]
    events: tuple[StepEvent, ...]


class ContinuousBatchEngine:
    """Fair round-robin token-budget execution over independent requests.

    Preemption discards a victim's physical KV and later replays its committed
    tokens from position zero, without losing already-generated output.
    Version-fenced scheduled work is skipped when an earlier item evicts it.
    """

    def __init__(
        self,
        model: ReferenceGPT2 | ReferenceQwen3,
        cache: PagedKVStore,
        *,
        max_num_scheduled_tokens: int = 32,
        max_num_seqs: int = 8,
        eos_token_id: int | None = None,
        model_fingerprint: str = "stage03",
    ) -> None:
        if not model_fingerprint:
            raise ValueError("model_fingerprint must be nonempty")
        heads = (
            model.config.num_kv_heads if isinstance(model, ReferenceQwen3)
            else model.config.num_heads
        )
        dim = (
            model.config.head_dim if isinstance(model, ReferenceQwen3)
            else model.config.hidden_size // model.config.num_heads
        )
        param = next(model.parameters())
        if (
            (cache.num_layers, cache.kv_heads, cache.head_dim)
            != (model.config.num_layers, heads, dim)
            or cache.dtype != param.dtype
            or cache.device != param.device
        ):
            raise ValueError("physical KV cache and model geometry/device mismatch")
        self.model = model.eval()
        self.cache = cache
        self.scheduler = UnifiedScheduler(max_num_scheduled_tokens, max_num_seqs)
        self.model_fingerprint = model_fingerprint
        self.eos_token_id = eos_token_id
        self._states: dict[str, _Sequence] = {}
        self._results: dict[str, GenerationResult] = {}
        self.history: list[EngineStep] = []
        self.preemption_count = 0

    @property
    def pending(self) -> int:
        return self.scheduler.pending

    @property
    def results(self) -> dict[str, GenerationResult]:
        return dict(self._results)

    def add_request(
        self, request_id: str, prompt_token_ids: tuple[int, ...],
        sampling: SamplingParams, *, stop_token_ids: tuple[int, ...] = (),
    ) -> None:
        if request_id in self._states:
            raise ValueError("duplicate request ID including completed requests")
        if any(token < 0 for token in stop_token_ids):
            raise ValueError("stop token IDs must be nonnegative")
        if len(prompt_token_ids) > self.model.config.max_positions:
            raise ValueError("prompt exceeds model context")
        request = Request(
            request_id, prompt_token_ids, sampling, self.model_fingerprint
        )
        device = next(self.model.parameters()).device
        rng = torch.Generator(device=device).manual_seed(sampling.seed or 0)
        self._states[request_id] = _Sequence(request, rng, stop_token_ids)
        self.scheduler.add(request_id)

    def cancel(self, request_id: str) -> bool:
        state = self._states.get(request_id)
        if state is None or state.status in ("cancelled", "failed", "finished"):
            return False
        self._terminal(state, "cancelled")
        return True

    def _terminal(self, state: _Sequence, reason: str, error: str | None = None) -> None:
        rid = state.request.request_id
        self.cache.release(rid)
        self.scheduler.remove(rid)
        if reason == "cancelled":
            state.status = "cancelled"
        elif reason == "failed":
            state.status = "failed"
        else:
            state.status = "finished"
        state.error = error
        state.last_logits = None
        state.version += 1
        self._results[rid] = GenerationResult(
            request_id=rid,
            prompt_token_ids=state.request.prompt_token_ids,
            generated_token_ids=tuple(state.generated),
            finish_reason=reason,
        )

    def _preempt_one(self, exclude: str) -> _Sequence | None:
        candidates = [
            s for s in self._states.values()
            if s.request.request_id != exclude
            and s.status in ("waiting", "running")
            and s.computed > 0
        ]
        if not candidates:
            return None
        # Deterministic victim: most physical pages, tie by submission order.
        victim = max(
            candidates,
            key=lambda s: len(self.cache.manager.table(s.request.request_id).blocks),
        )
        self.cache.release(victim.request.request_id)
        victim.computed = 0
        victim.last_logits = None
        victim.version += 1
        victim.preemptions += 1
        victim.status = "waiting"
        self.preemption_count += 1
        return victim

    def _publish(self, state: _Sequence, kv: object, events: list[StepEvent]) -> None:
        while True:
            try:
                self.cache.write(state.request.request_id, kv)
                return
            except KVCapacityError:
                victim = self._preempt_one(state.request.request_id)
                if victim is None:
                    raise
                events.append(StepEvent(victim.request.request_id, "preempt", 0))

    @torch.inference_mode()
    def step(self) -> EngineStep:
        if not self.pending:
            raise RuntimeError("no active requests")
        views = {
            rid: RequestView(rid, len(s.context), s.computed, s.version)
            for rid, s in self._states.items()
            if s.status in ("waiting", "running")
        }
        batch: ScheduledBatch = self.scheduler.plan(views)
        if not batch.work:
            raise RuntimeError("scheduler made no progress")
        events: list[StepEvent] = []
        for work in batch.work:
            state = self._states[work.request_id]
            if state.version != work.version or state.status not in ("waiting", "running"):
                events.append(StepEvent(work.request_id, "skip", 0))
                continue
            state.status = "running"
            try:
                self._execute(work, state, events)
            except Exception as error:
                # Do not poison other requests; release committed/pending pages.
                self._terminal(state, "failed", f"{type(error).__name__}: {error}")
                events.append(
                    StepEvent(work.request_id, "finish", 0, reason="failed")
                )
        report = EngineStep(
            batch.step_id, batch.token_budget, batch.used_tokens,
            tuple(w.request_id for w in batch.work), tuple(events),
        )
        self.history.append(report)
        return report

    def _execute(
        self, work: ScheduledWork, state: _Sequence, events: list[StepEvent],
    ) -> None:
        rid = work.request_id
        device = next(self.model.parameters()).device
        if work.kind == "prefill":
            context = state.context
            fragment = context[
                state.computed:state.computed + work.token_count
            ]
            previous = self.cache.read(rid) if state.computed else None
            tokens = torch.tensor([fragment], dtype=torch.long, device=device)
            logits, kv = self.model(tokens, cache=previous)
            self._publish(state, kv, events)
            state.computed += len(fragment)
            state.last_logits = logits[0, -1]
            state.version += 1
            events.append(StepEvent(rid, "prefill", len(fragment)))
            return

        if state.computed != len(state.context) or state.last_logits is None:
            raise RuntimeError("decode without fully computed prefix")
        previous_rng = state.rng.get_state()
        token = sample_next(
            state.last_logits, state.request.sampling, state.rng
        )
        reason: str | None = None
        if token == self.eos_token_id:
            reason = "eos"
        elif token in state.stop_ids:
            reason = "stop_token"
        elif len(state.generated) + 1 >= state.request.sampling.max_new_tokens:
            reason = "length"
        elif len(state.context) + 1 > self.model.config.max_positions:
            reason = "context_limit"
        if reason is None:
            try:
                previous = self.cache.read(rid)
                logits, kv = self.model(
                    torch.tensor([[token]], dtype=torch.long, device=device),
                    cache=previous,
                )
                self._publish(state, kv, events)
            except Exception:
                state.rng.set_state(previous_rng)
                raise
            state.computed += 1
            state.last_logits = logits[0, -1]
        state.generated.append(token)
        state.version += 1
        events.append(StepEvent(rid, "decode", 1, token_id=token))
        if reason is not None:
            self._terminal(state, reason)
            events.append(StepEvent(rid, "finish", 0, reason=reason))

    def run(self, *, max_steps: int = 10000) -> dict[str, GenerationResult]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        for _ in range(max_steps):
            if not self.pending:
                return self.results
            self.step()
        if self.pending:
            raise RuntimeError("max_steps exceeded; scheduler may be stuck")
        return self.results

    def assert_consistent(self) -> None:
        self.cache.assert_consistent()
        if self.scheduler.pending != sum(
            s.status in ("waiting", "running") for s in self._states.values()
        ):
            raise AssertionError("active scheduler and execution slots disagree")
        for rid, state in self._states.items():
            if state.status in ("finished", "cancelled", "failed"):
                continue
            if state.computed:
                if self.cache.manager.table(rid).num_tokens != state.computed:
                    raise AssertionError("computed cursor and physical KV mismatch")
