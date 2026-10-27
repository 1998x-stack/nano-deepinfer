# Stage 04 — Chunked prefill & prefix cache

**Status:** NOT STARTED. **Dependencies:** see [ROADMAP](../../ROADMAP.md). This document is a work plan, not implementation evidence.

## Engineering tasks

- [ ] Write a focused ADR/RFC covering problem constraints, alternatives and correctness invariants.
- [ ] Implement Chunked prefill plan; content hash+model fingerprint+tenant isolation prefix index.
- [ ] Add unit, property-based, fault-path and reference-oracle tests before optimization.
- [ ] Implement safe fallback and feature flag; document supported hardware/model boundaries.
- [ ] Run controlled A/B comparison on pinned model, source commit, workload and hardware.
- [ ] Update documentation, CI and `docs/status.md` with measured evidence.

## Exit gate

Shared-prefix outputs equal uncached reference; no cross-tenant/cache-version contamination; release v0.1.

## Deliverables

Working code with tests, a reproducible example, source notes, benchmark raw artifacts, and a PR containing risks and rollback plan.

**Do not start until earlier gates are reviewed; no performance claims without measurements.**

