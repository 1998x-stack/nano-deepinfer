# ADR 0001 — Scope and non-goals

**Status:** Accepted (Stage 00). **Date:** 2026-10-27.

Implement a self-contained tiny inference engine rather than a DeepInfer fork. First target is single-request small dense model, then scheduling/memory and optional hardware accelerators. `main` always reflects last verified progress. CPU-only engineering paths must remain testable. A full V4.1-Flash deployment, proprietary distributed cache, DeepSeek production throughput or vendor-specific SM100/Ascend kernels are not v1.0 requirements.

**Trade-offs:** More implementation work, but clear ownership, less vendor dependence and stronger teaching value. Any future vendor code import must be separately reviewed and credited.

