# Stage 01 — Minimal Dense reference model & inference loop

**Status:** NOT STARTED. **Estimated:** 1–2 weeks. **Prerequisite:** Stage 00 baseline/status approved.

## Scope and work packages

- [ ] S01-01 Config, SamplingParams, Request, SequenceState, tokenizer adapter validation.
- [ ] S01-02 CPU toy CausalLM + PyTorch eager model adapter contract.
- [ ] S01-03 Qwen3-0.6B checkpoint adapter, correct ROPE/attention and tokenizer options (trust_remote_code=False).
- [ ] S01-04 Separate prefill logits and single-token decode, sampler greedy/temperature/top-k/top-p with seed, EOS, stop tokens and max-new-tokens.
- [ ] S01-05 Oracle comparisons vs **identical immutable HF checkpoint and tokenizer** for logits and token IDs.
- [ ] S01-06 Tests: zero/long/invalid prompts, EOS, deterministic seeds, cancellation, resource reclamation, wrong dtype and logits tolerance.
- [ ] S01-07 Produce actual timing and memory metrics or explicitly mark unsupported.

## Acceptance

Greedy tokens match HF at pinned checkpoint/config, logits within documented dtype tolerances, sampling tests independently validated, no continuous batching or Paged KV claims (Stage 02–03). CPU oracle is independently implemented; model execution must not call vLLM.

**Outputs:** `nano_deepinfer.LLM.generate`, `examples/basic.py`, `tests/oracle/` plus benchmark provenance and `docs/status.md` update.

