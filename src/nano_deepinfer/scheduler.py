"""Stage 03 deterministic unified token-budget scheduler (CPU control plane).

Plans prefill and decode work in one token budget, with bounded FCFS
round-robin fairness. This is *logical* continuous batching; the Stage 01
model backend still executes batch size one for each planned request.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

Kind = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class RequestView:
    request_id: str
    total_context_tokens: int
    computed_tokens: int
    version: int
    resident: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or self.version < 0:
            raise ValueError("invalid request identity or version")
        if not 0 <= self.computed_tokens <= self.total_context_tokens:
            raise ValueError("invalid computed-token cursor")


@dataclass(frozen=True, slots=True)
class ScheduledWork:
    request_id: str
    version: int
    kind: Kind
    token_count: int
    input_offset: int


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    step_id: int
    token_budget: int
    used_tokens: int
    work: tuple[ScheduledWork, ...]

    def __post_init__(self) -> None:
        if self.step_id < 0 or not 0 <= self.used_tokens <= self.token_budget:
            raise ValueError("invalid scheduling token accounting")
        if sum(item.token_count for item in self.work) != self.used_tokens:
            raise ValueError("work does not match scheduled token count")
        if len({item.request_id for item in self.work}) != len(self.work):
            raise ValueError("request scheduled twice in one batch")


class UnifiedScheduler:
    """Fair admission planner. Does not hold KV blocks or execute models."""

    def __init__(self, max_num_scheduled_tokens: int, max_num_seqs: int) -> None:
        if max_num_scheduled_tokens < 1 or max_num_seqs < 1:
            raise ValueError("scheduler budgets must be positive")
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.max_num_seqs = max_num_seqs
        self._queue: deque[str] = deque()
        self._live: set[str] = set()
        self._step = 0

    def add(self, request_id: str) -> None:
        if not request_id or request_id in self._live:
            raise ValueError("empty or duplicate active request ID")
        self._live.add(request_id)
        self._queue.append(request_id)

    def remove(self, request_id: str) -> bool:
        if request_id not in self._live:
            return False
        self._live.remove(request_id)
        self._queue.remove(request_id)
        return True

    @property
    def pending(self) -> int:
        return len(self._live)

    def plan(self, active: dict[str, RequestView]) -> ScheduledBatch:
        """One work item per request, max-sequences and token-budget enforced.

        A prefill chunk can be smaller than its remaining context, including
        reconstruction of an evicted sequence; this is a correctness-only
        chunking fallback. Optimized chunked-prefill policy is Stage 04.
        """
        if set(active) != self._live:
            raise ValueError("scheduler state and active views disagree")
        available = self.max_num_scheduled_tokens
        selected: list[ScheduledWork] = []
        # Visit every entry at most once; picked and unpicked entries rotate
        # together, without duplicating an ID within one scheduling step.
        order = list(self._queue)
        # Prefer resident requests so new arrivals cannot continually evict
        # unfinished decoders under severe page pressure. This is bounded
        # residency priority, not absolute global FCFS completion order.
        order.sort(key=lambda rid: not active[rid].resident)
        for rid in order:
            if available == 0 or len(selected) == self.max_num_seqs:
                continue
            view = active[rid]
            remaining = view.total_context_tokens - view.computed_tokens
            if remaining:
                count = min(remaining, available)
                selected.append(ScheduledWork(rid, view.version, "prefill", count,
                                              view.computed_tokens))
            else:
                count = 1
                selected.append(ScheduledWork(rid, view.version, "decode", count,
                                              view.computed_tokens))
            available -= count
        if selected:
            # Rotate the next scheduling turn past serviced requests, not merely
            # a full traversal back to the original FCFS queue position.
            self._queue.rotate(-len(selected))
        batch = ScheduledBatch(
            self._step, self.max_num_scheduled_tokens,
            self.max_num_scheduled_tokens - available, tuple(selected),
        )
        self._step += 1
        return batch
