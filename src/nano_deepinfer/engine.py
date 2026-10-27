# mypy: ignore-errors
"""Stage 01 synchronous single-request engine; no continuous batching or paged KV."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import torch
from torch import Tensor

from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.dense_gpt2 import ModelKV, ReferenceGPT2
from nano_deepinfer.dense_qwen3 import ReferenceQwen3
from nano_deepinfer.foundation import (
    Request,
    SamplingParams,
    StepPlan,
    StepResult,
    validate_step_result,
)


@dataclass(slots=True)
class SequenceState:
    request: Request
    generated: list[int] = field(default_factory=list)
    cache: ModelKV | None = None
    computed_tokens: int = 0
    finished: bool = False


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    finish_reason: str


def sample_next(logits: Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    """Greedy at temperature=0, otherwise correctly ordered top-k then nucleus."""
    if logits.ndim != 1:
        raise ValueError("logits must be a vocabulary vector")
    if params.temperature == 0.0:
        return int(torch.argmax(logits).item())
    scaled = logits.float() / params.temperature
    if params.top_k is not None and params.top_k < scaled.shape[0]:
        cut = torch.topk(scaled, params.top_k).values[-1]
        scaled = scaled.masked_fill(scaled < cut, -torch.inf)
    if params.top_p < 1.0:
        descending, indices = torch.sort(scaled, descending=True)
        cum = torch.softmax(descending, dim=0).cumsum(0)
        remove = cum > params.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scaled = scaled.scatter(0, indices, descending.masked_fill(remove, -torch.inf))
    probs = torch.softmax(scaled, dim=0)
    if not torch.all(torch.isfinite(probs)):
        raise ValueError("non-finite probabilities")
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


class LLMEngine:
    """Reference-only model runner. Accepts token IDs to avoid hiding tokenizer state."""

    def __init__(
        self, model: ReferenceGPT2 | ReferenceQwen3, model_fingerprint: str = "synthetic", eos_token_id: int | None = None,
        paged_cache: PagedKVStore | None = None,
    ) -> None:
        if not model_fingerprint:
            raise ValueError("model_fingerprint required")
        self.model = model.eval()
        self.model_fingerprint = model_fingerprint
        self.eos_token_id = eos_token_id
        if paged_cache is not None:
            cfg = model.config
            layers = cfg.num_layers
            heads = cfg.num_kv_heads if isinstance(model, ReferenceQwen3) else cfg.num_heads
            dim = cfg.head_dim if isinstance(model, ReferenceQwen3) else cfg.hidden_size // cfg.num_heads
            if (paged_cache.num_layers, paged_cache.kv_heads, paged_cache.head_dim) != (layers, heads, dim):
                raise ValueError("paged cache dimensions do not match model")
            parameter = next(model.parameters())
            if paged_cache.dtype != parameter.dtype or paged_cache.device != parameter.device:
                raise ValueError("paged cache device/dtype do not match model")
        self.paged_cache = paged_cache

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: tuple[int, ...],
        sampling: SamplingParams,
        request_id: str | None = None,
        stop_token_ids: tuple[int, ...] = (),
    ) -> GenerationResult:
        if any(token < 0 for token in stop_token_ids):
            raise ValueError("stop token IDs must be nonnegative")
        request = Request(request_id or uuid4().hex, prompt_token_ids, sampling, self.model_fingerprint)
        state = SequenceState(request=request)
        device = next(self.model.parameters()).device
        rng = torch.Generator(device=device).manual_seed(sampling.seed or 0)
        if len(prompt_token_ids) > self.model.config.max_positions:
            raise ValueError("prompt exceeds model context")
        try:
            inputs = torch.tensor([list(prompt_token_ids)], device=device, dtype=torch.long)
            logits, cache = self.model(inputs)
            if self.paged_cache is not None:
                self.paged_cache.write(request.request_id, cache)
            else:
                state.cache = cache
            state.computed_tokens = len(prompt_token_ids)
            last_logits = logits[0, -1, :]
            reason = "length"
            for step_id in range(sampling.max_new_tokens):
                plan = StepPlan(
                    step_id=step_id,
                    request_versions=((request.request_id, step_id),),
                    token_budget=1,
                )
                token = sample_next(last_logits, sampling, rng)
                validate_step_result(
                    plan, StepResult(
                        step_id=step_id, request_versions=plan.request_versions, worker_ack=True
                    )
                )
                state.generated.append(token)
                if token == self.eos_token_id:
                    reason = "eos"
                    break
                if token in stop_token_ids:
                    reason = "stop_token"
                    break
                if step_id + 1 == sampling.max_new_tokens:
                    break
                if state.computed_tokens + 1 > self.model.config.max_positions:
                    reason = "context_limit"
                    break
                next_ids = torch.tensor([[token]], device=device, dtype=torch.long)
                active_cache = (
                    self.paged_cache.read(request.request_id)
                    if self.paged_cache is not None else state.cache
                )
                next_logits, updated_cache = self.model(next_ids, cache=active_cache)
                if self.paged_cache is not None:
                    self.paged_cache.write(request.request_id, updated_cache)
                else:
                    state.cache = updated_cache
                state.computed_tokens += 1
                last_logits = next_logits[0, -1, :]
            state.finished = True
            return GenerationResult(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                generated_token_ids=tuple(state.generated),
                finish_reason=reason,
            )
        finally:
            if self.paged_cache is not None:
                # Both success and exception paths must release physical pages.
                self.paged_cache.release(request.request_id)
