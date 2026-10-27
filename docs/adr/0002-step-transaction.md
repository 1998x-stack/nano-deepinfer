# ADR 0002 — Schedule / execute / commit separation

**Status:** Accepted as design, not implemented. **Date:** 2026-10-27.

Scheduler creates an immutable plan with a monotonically increasing step ID and per-request version. Executor returns immutable results. Committer rejects stale versions and duplicate request/step pairs before exposing tokens; resource releases are guarded by worker completion acknowledgement. Stage 01 uses a synchronous path honoring these boundaries; Stage 06 introduces overlapping execution.

**Risk mitigated:** out-of-order async steps, speculative rollback corruption, double free and canceled-request output. **Tests later:** reordering, duplicated results, dropped worker acknowledgements, forced exception, cancellation after dispatch.

