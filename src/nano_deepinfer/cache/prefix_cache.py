"""Stage 04 block-aligned immutable KV prefix index.

A cache entry pins the physical pages (via BlockManager refs), not a
second copy of the entire KV. An attached request adds its own references.
Eviction drops only cache pins; live request owners remain valid.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor  # type: ignore[import-not-found]

from nano_deepinfer.cache.block_manager import (
    BlockHandle,
    BlockManager,
    BlockTable,
    KVStateError,
)


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Explicit isolation boundary; caller MUST identify actual model weights."""

    model_revision: str
    tokenizer_revision: str
    positional_config: str
    tenant_id: str
    dtype: str
    adapter_revision: str = "none"
    cache_schema: str = "stage04-v1"

    def __post_init__(self) -> None:
        values = (
            self.model_revision, self.tokenizer_revision, self.positional_config,
            self.tenant_id, self.dtype, self.adapter_revision, self.cache_schema,
        )
        if any(not value.strip() for value in values):
            raise ValueError("prefix cache namespace fields must be nonempty")

    def key_for(self, tokens: tuple[int, ...]) -> str:
        if not tokens or any(token < 0 for token in tokens):
            raise ValueError("prefix tokens must be nonempty nonnegative IDs")
        # Canonical typed length-delimited JSON makes tuple token boundaries explicit.
        payload = json.dumps(
            {"scope": (
                self.cache_schema, self.model_revision, self.tokenizer_revision,
                self.positional_config, self.tenant_id, self.dtype,
                self.adapter_revision,
            ), "tokens": tokens},
            ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PrefixHit:
    matched_tokens: int
    last_logits: Tensor
    key: str


@dataclass(slots=True)
class _Entry:
    tokens: tuple[int, ...]
    blocks: tuple[BlockHandle, ...]
    last_logits: Tensor
    namespace: CacheNamespace


class PrefixCache:
    """LRU index owning page pins. Capacity counts pinned page references."""

    def __init__(self, manager: BlockManager, max_cached_blocks: int) -> None:
        if max_cached_blocks < 1:
            raise ValueError("max_cached_blocks must be positive")
        self.manager = manager
        self.max_cached_blocks = max_cached_blocks
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _num_pinned(self) -> int:
        return sum(len(entry.blocks) for entry in self._entries.values())

    @property
    def cached_blocks(self) -> int:
        with self._lock:
            return self._num_pinned()

    @property
    def entries(self) -> int:
        with self._lock:
            return len(self._entries)

    def evict_one(self) -> bool:
        with self._lock:
            if not self._entries:
                return False
            _, entry = self._entries.popitem(last=False)
            self.manager.unpin(entry.blocks)
            self.evictions += 1
            return True

    def clear(self, tenant_id: str | None = None) -> int:
        """Remove cache pins, optionally only one tenant's index entries."""
        with self._lock:
            keys = [
                key for key, entry in self._entries.items()
                if tenant_id is None or entry.namespace.tenant_id == tenant_id
            ]
            for key in keys:
                entry = self._entries.pop(key)
                self.manager.unpin(entry.blocks)
                self.evictions += 1
            return len(keys)

    def publish(
        self,
        namespace: CacheNamespace,
        tokens: tuple[int, ...],
        table: BlockTable,
        last_logits: Tensor,
    ) -> bool:
        """Pin a completed aligned prefix; no mutation of an existing key."""
        if (
            len(tokens) < self.manager.block_size
            or len(tokens) % self.manager.block_size
            or table.num_tokens != len(tokens)
            or len(table.blocks) != len(tokens) // self.manager.block_size
            or len(table.blocks) > self.max_cached_blocks
        ):
            return False
        key = namespace.key_for(tokens)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return False
            self.manager.pin(table.blocks)
            try:
                # Snapshot must not alias a mutable model output tensor.
                owned_logits = last_logits.detach().clone()
                self._entries[key] = _Entry(
                    tokens, table.blocks, owned_logits, namespace
                )
                while self._num_pinned() > self.max_cached_blocks:
                    self.evict_one()
            except BaseException:
                if key in self._entries:
                    del self._entries[key]
                self.manager.unpin(table.blocks)
                raise
            return key in self._entries

    def attach_longest(
        self, request_id: str, namespace: CacheNamespace,
        prompt: tuple[int, ...],
    ) -> PrefixHit | None:
        """Attach longest *full-block* hit without copying KV page contents.

        Reject exact digest collisions via explicit full scope and token equality.
        No request state is created on miss; caller may create a new empty table.
        """
        with self._lock:
            aligned = len(prompt) // self.manager.block_size * self.manager.block_size
            for length in range(aligned, 0, -self.manager.block_size):
                tokens = prompt[:length]
                key = namespace.key_for(tokens)
                entry = self._entries.get(key)
                if (
                    entry is None or entry.tokens != tokens
                    or entry.namespace != namespace
                ):
                    continue
                try:
                    self.manager.attach_prefix(request_id, entry.blocks, length)
                except KVStateError:
                    # A matching namespace does not excuse invalid request IDs.
                    raise
                self._entries.move_to_end(key)
                self.hits += 1
                return PrefixHit(length, entry.last_logits.detach().clone(), key)
            self.misses += 1
            return None

    def assert_consistent(self) -> None:
        with self._lock:
            if self._num_pinned() > self.max_cached_blocks:
                raise AssertionError("prefix index exceeded its capacity")
            for key, entry in self._entries.items():
                if key != entry.namespace.key_for(entry.tokens):
                    raise AssertionError("cache namespace/token digest mismatch")
                if len(entry.blocks) * self.manager.block_size != len(entry.tokens):
                    raise AssertionError("partial block entered shared prefix")
                if any(not self.manager.is_live(h) for h in entry.blocks):
                    raise AssertionError("prefix index points to an evicted block")
            self.manager.assert_consistent()
