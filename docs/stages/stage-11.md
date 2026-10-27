# Stage 11 — Experimental P/D disaggregation

**Status:** NOT STARTED. **Dependencies:** see [ROADMAP](../../ROADMAP.md). This document is a work plan, not implementation evidence.

## Engineering tasks

- [ ] Write a focused ADR/RFC covering problem constraints, alternatives and correctness invariants.
- [ ] Implement Separate prefill/decode workers, typed KV handoff, ACK and failure handling.
- [ ] Add unit, property-based, fault-path and reference-oracle tests before optimization.
- [ ] Implement safe fallback and feature flag; document supported hardware/model boundaries.
- [ ] Run controlled A/B comparison on pinned model, source commit, workload and hardware.
- [ ] Update documentation, CI and `docs/status.md` with measured evidence.

## Exit gate

Reference equivalence and end-to-end TTFT/TPOT/network cost model; not a release blocker.

## Deliverables

Working code with tests, a reproducible example, source notes, benchmark raw artifacts, and a PR containing risks and rollback plan.

**Do not start until earlier gates are reviewed; no performance claims without measurements.**

