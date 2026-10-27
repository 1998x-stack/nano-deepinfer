# ADR 0004 — Transactional physical KV paging in Stage 02

**Status:** Implemented as synchronous CPU-first reference, 2026-10-27.

## Decision

Implement an independent, testable block allocator and physical Tensor page storage. Use request-owned immutable `BlockHandle(block_id, generation)` handles; a `BlockTable` is a committed snapshot of logical tokens mapped to physical blocks. Each request permits one outstanding `Reservation`. Storage writes happen to privately allocated pages; only after all writes succeed is the updated logical table committed.

**Atomicity:** reserve all required blocks before modifying shared state; partial pages are copy-on-write (COW). On write failure abort only the newly allocated pages, preserving the last committed prefix. Out-of-capacity reservation never mutates the old table. On cancellation or finish, release all owned and pending blocks; old handles are invalidated by generations. The current allocator does **not share** pages between requests; refcount is exclusively 0 or 1.

**Conservative memory consequence:** appending to a partially occupied page requires one temporarily free page, even when the token fits within its existing physical page. A full pool can reject such an append; this is a conscious correctness-first trade-off pending measured in-place transactional optimization.

## Integration

`PagedKVStore` allocates physical K/V tensors sized `[num_blocks, kv_heads, block_size, head_dim]` per layer. A synchronous `LLMEngine(..., paged_cache=store)` path materializes dense cache from physical pages before calling the Stage 01 attention model, and stores resulting full per-layer KV via transactional page writes. `finally` releases the request. **It is a real paged storage implementation, but not a paged attention kernel and not a performance speedup.**

## Alternatives rejected

- Wrap vLLM: conflicts with independent-reimplementation constraint.
- Rewrite entire attention kernel in Stage 02: conflates memory correctness and GPU-kernel optimization.
- Mutate a partially filled committed page directly: could corrupt visible KV if a failed/canceled step overwrites data.
- Enable shared prefix hashing in Stage 02: would require careful tenant/model isolation and copy-on-write refcounts; deferred to Stage 04.

## Independent verification

Deterministic randomized owner/length-oracle property tests; stale reservation and generation tests; concurrent create/grow/release; OOM; injected mid-copy data failure; model-token equality with dense reference and actual pinned HF tiny GPT-2 and Qwen3 checkpoints. See [Stage 02](../stages/stage-02.md).

## Deferred

PagedAttention (GPU direct physical page lookup); Prefix caching; multisequence scheduling; async GPU writes and completion fences; model-specialized mixed KV shapes and SWA; GPU throughput. Those are not delivered in Stage 02.
