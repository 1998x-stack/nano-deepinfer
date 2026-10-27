# RFC 0001: Engine core contracts and reference-first execution

**Status:** Accepted for Stage 00 design; implementations pending. **Date:** 2026-10-27

## Decision

Build a small **independently authored** engine in Python 3.11+, separating (A) request lifecycle (B) unified token budget scheduler (C) cache manager (D) runner (E) sampler (F) idempotent output committer. The compiler/kernel/MoE/distributed paths are independently enabled feature gates. vLLM and DeepInfer are research references, not hard runtime dependencies.

## Proposed typed protocol

```python
@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    model_fingerprint: str

@dataclass(frozen=True)
class StepPlan:
    step_id: int
    request_versions: tuple[tuple[str, int], ...]
    scheduled_tokens: tuple[tuple[str, tuple[int, ...]], ...]
    reservation_ids: tuple[int, ...]

@dataclass(frozen=True)
class StepResult:
    step_id: int
    request_versions: tuple[tuple[str, int], ...]
    output_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    worker_ack: bool
```

The above is **a contract proposal**, not a claim that all models currently run.

## Formal invariants

1. `request_id` nonempty and unique per active request; prompt tokens nonnegative integers.
2. `max_new_tokens>=1`, `temperature>=0`, `top_k>0` if set, `0<top_p<=1`.
3. `step_id>=0`; request versions never decrease; a stale result cannot be committed.
4. Exactly-once commit means at most one visible commit for `(request_id, step_id)`; duplicates safely ignored or rejected.
5. KV reservation is exclusive until published; release after cancellation/OOM must not expose stale contents.
6. Batch output ordering follows request identity, not incidental GPU tensor ordering.
7. Greedy oracle is token-exact only for equivalent model, tokenizer, dtype and numerics; sampling oracle requires distribution tests.

## Alternatives rejected

- Reuse vLLM as execution backend: fails independence and research aims.
- Copy DeepInfer and delete cluster-specific code: model/hardware specialization remains and adds provenance burden.
- Scheduler emits tokens directly: breaks async transaction semantics and rollback safety.

## Verification plan

Stage 00: schema/property tests on contracts and synthetic determinism; no inference claims. Stage 01: compare logits and token IDs against same-pinned-weights Transformers implementation; record expected numerical tolerance. Stage 02: randomized alloc/free/abort property tests. Stage 03: deterministic scheduler trace replay. Stage 06+: stale, canceled, duplicate output races and GPU fault injection.

## Risk register

Lack of model checkpoint / GPU limits accuracy and performance comparison; unoptimized reference must remain available. CPU synthetic smoke does **not** substitute for HF correctness testing. See [status](../status.md).

