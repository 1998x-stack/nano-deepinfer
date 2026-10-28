# ADR 0006 — Immutable, namespace-isolated shared prefix KV blocks

**Status:** Stage 04 CPU prototype; hardware validation remains open.

## Design decision

Cache only **fully populated, block-aligned** prefixes. A `CacheNamespace` explicitly binds model weight revision, tokenizer revision, positional configuration, tenant ID, KV dtype, adapter revision and cache schema. A canonical serialized content digest uses SHA-256 on the full prefix token sequence and all namespace fields. A digest match must also pass exact namespace/token equality before physical pages are attached. The caller must supply trustworthy immutable revision identifiers; metadata supplied by an untrusted client is **not** sufficient to assert model compatibility.

Each indexed entry owns **pins** on the original Stage 02 physical K/V `BlockHandle`s. A cache hit increments each shared block's reference count by attaching an existing aligned `BlockTable` to a new request, without copying the K/V payload. Releasing an in-flight request does not evict pinned pages; evicting an LRU cache entry does not invalidate pages used by other requests. A partial last page is never shared; its future append uses the existing COW transaction.

A cached last-token logits vector is snapshotted for the exact aligned prompt, allowing immediate decode without rerunning that prompt. For a longer prompt sharing the cached aligned prefix, the remaining suffix runs through the ordinary independent GPT-2/Qwen3 prefill path and produces fresh logits. The default engine with `max_cached_blocks=0` preserves Stage 03 behavior.

## Integration

- `BlockManager.pin/unpin/attach_prefix`: physical block lifetime and generation fencing. `assert_consistent` audits all request references, pending allocations and cache-owned pins.
- `PrefixCache`: longest full-block prefix search, LRU index, per-tenant invalidation, cache-owned pin capacity and hash collision defense.
- `PagedKVStore`: optional prefix index, transparent LRU eviction on allocation pressure and physical readback.
- `ContinuousBatchEngine`: opt-in explicit namespace, restore on new request or recompute, last-logits reuse, publication after aligned prompt prefill, hit/tokens counters.
- `UnifiedScheduler`: configurable `max_prefill_chunk_size` independent of token-budget and request-slot limits.

## Safety and fault invariants

1. A stale block generation cannot be pinned/attached; unbalanced pin release errors rather than double-freeing.
2. Uncommitted or partial physical pages never enter the shared index.
3. Tenant/model/tokenizer/position/dtype/adapter/cache-schema changes must produce different keys.
4. LRU eviction removes **only** its own pin; in-flight consumers retain KV data and can append safely.
5. Physical cache capacity can be reclaimed by evicting idle prefix pins before failing a request.
6. Full/partially-cached GPT-2 and Qwen3 outputs must match the same-weights uncached reference exactly for greedy and per-request seeded sampling.

## Scope boundary

Current cache indexes only full-block aligned **completed prompt prefixes**. It does not cache arbitrary unaligned prefixes, intermediate prompt chunks, mixed model layers, SWA state, or other model-family state. No GPU PagedAttention exists. Logical multi-request work still executes batch-size-one model forwards in series. KV materialization and prefix immutability checks may be expensive; even where tokens are skipped, production speedup is not established.

The Stage 00 matching-GPU vLLM baseline and full trained Qwen3-0.6B matrix are **NOT RUN**. No final v0.1 release tag should be created until the original gates or an explicit reviewed waiver are satisfied.
