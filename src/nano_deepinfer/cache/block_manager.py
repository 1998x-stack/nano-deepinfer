"""Stage 02: CPU-only transactional paged KV block ownership.

Own code. No vLLM runtime dependency. No prefix sharing (Stage 04), no
scheduling (Stage 03), and no CUDA-native PagedAttention (Stage 05).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil
from threading import RLock


class KVCapacityError(RuntimeError):
    """Not enough *free* blocks to reserve atomically."""


class KVStateError(RuntimeError):
    """Invalid owner, stale reservation, or inconsistent lifecycle."""


@dataclass(frozen=True, slots=True)
class BlockHandle:
    block_id: int
    generation: int


@dataclass(frozen=True, slots=True)
class BlockTable:
    request_id: str
    num_tokens: int
    blocks: tuple[BlockHandle, ...]


@dataclass(frozen=True, slots=True)
class Reservation:
    request_id: str
    reservation_id: int
    target_tokens: int
    original: tuple[BlockHandle, ...]
    proposed: tuple[BlockHandle, ...]
    fresh: tuple[BlockHandle, ...]


@dataclass(frozen=True, slots=True)
class PoolStats:
    capacity: int
    used: int
    free: int
    owners: int
    pending: int


@dataclass(slots=True)
class _Record:
    num_tokens: int = 0
    blocks: list[BlockHandle] = field(default_factory=list)
    pending: Reservation | None = None


class BlockManager:
    """Fixed-size physical page allocator with per-request ownership.

    Commit only after the data plane successfully writes all new pages.
    A partially occupied last block is copied-on-write before appending
    tokens. Failed writes can then abort without corrupting committed data.

    For now ref_count is 0 or 1: prefix sharing is a separate future stage.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._lock = RLock()
        self._free: deque[int] = deque(range(num_blocks))
        self._generations = [0] * num_blocks
        self._refs = [0] * num_blocks
        self._owners: dict[str, _Record] = {}
        self._next_id = 0

    def create(self, request_id: str) -> BlockTable:
        if not request_id or not request_id.strip():
            raise ValueError("request_id must be nonempty")
        with self._lock:
            if request_id in self._owners:
                raise KVStateError("request already exists")
            self._owners[request_id] = _Record()
            return BlockTable(request_id, 0, ())

    def table(self, request_id: str) -> BlockTable:
        with self._lock:
            r = self._owner(request_id)
            return BlockTable(request_id, r.num_tokens, tuple(r.blocks))

    def _owner(self, request_id: str) -> _Record:
        try:
            return self._owners[request_id]
        except KeyError:
            raise KVStateError("unknown request") from None

    def _take(self) -> BlockHandle:
        block_id = self._free.popleft()
        if self._refs[block_id] != 0:
            raise AssertionError("free block is referenced")
        self._refs[block_id] = 1
        self._generations[block_id] += 1
        return BlockHandle(block_id, self._generations[block_id])

    def _put(self, handle: BlockHandle) -> None:
        if not self.is_live(handle):
            raise KVStateError("stale or already released block")
        if self._refs[handle.block_id] != 1:
            raise AssertionError("Stage 02 never shares blocks")
        self._refs[handle.block_id] = 0
        self._free.append(handle.block_id)

    def is_live(self, handle: BlockHandle) -> bool:
        with self._lock:
            return (
                0 <= handle.block_id < self.num_blocks
                and self._generations[handle.block_id] == handle.generation
                and self._refs[handle.block_id] == 1
            )

    def begin(self, request_id: str, target_tokens: int) -> Reservation:
        """Atomically allocate pages for a new total length (not a delta)."""
        with self._lock:
            record = self._owner(request_id)
            if record.pending is not None:
                raise KVStateError("request has an uncommitted reservation")
            if target_tokens < record.num_tokens:
                raise ValueError("cannot shrink via begin")
            if target_tokens < 0:
                raise ValueError("target_tokens must be nonnegative")
            old = tuple(record.blocks)
            old_count = len(old)
            count = ceil(target_tokens / self.block_size)
            grown = target_tokens > record.num_tokens
            cow = bool(grown and record.num_tokens % self.block_size and old_count)
            required = count - old_count + int(cow)
            if len(self._free) < required:
                raise KVCapacityError(
                    f"need {required} blocks, only {len(self._free)} free"
                )
            fresh = tuple(self._take() for _ in range(required))
            if cow:
                proposed = old[:-1] + fresh
            else:
                proposed = old + fresh
            if len(proposed) != count:
                raise AssertionError("reservation page count mismatch")
            self._next_id += 1
            reservation = Reservation(
                request_id, self._next_id, target_tokens, old, proposed, fresh
            )
            record.pending = reservation
            return reservation

    def _pending(self, reservation: Reservation) -> _Record:
        record = self._owner(reservation.request_id)
        if record.pending is not reservation:
            raise KVStateError("stale or foreign reservation")
        return record

    def commit(self, reservation: Reservation) -> BlockTable:
        with self._lock:
            record = self._pending(reservation)
            if record.blocks and reservation.proposed[:len(record.blocks)] != tuple(record.blocks):
                # COW affects exactly the last old page, never prior committed pages.
                if (
                    reservation.proposed[:len(record.blocks) - 1]
                    != tuple(record.blocks[:-1])
                    or record.num_tokens % self.block_size == 0
                ):
                    raise AssertionError("unexpected page remapping")
                self._put(record.blocks[-1])
            record.blocks = list(reservation.proposed)
            record.num_tokens = reservation.target_tokens
            record.pending = None
            return BlockTable(reservation.request_id, record.num_tokens, tuple(record.blocks))

    def abort(self, reservation: Reservation) -> None:
        with self._lock:
            record = self._pending(reservation)
            for block in reservation.fresh:
                self._put(block)
            record.pending = None

    def release(self, request_id: str) -> bool:
        """Idempotent cancel/finish; stale block handles never become live again."""
        with self._lock:
            record = self._owners.pop(request_id, None)
            if record is None:
                return False
            if record.pending is not None:
                for block in record.pending.fresh:
                    self._put(block)
            for block in record.blocks:
                self._put(block)
            return True

    def stats(self) -> PoolStats:
        with self._lock:
            return PoolStats(
                self.num_blocks,
                sum(self._refs),
                len(self._free),
                len(self._owners),
                sum(r.pending is not None for r in self._owners.values()),
            )

    def assert_consistent(self) -> None:
        """Independent ownership oracle; deliberately O(capacity)."""
        with self._lock:
            free = list(self._free)
            if len(free) != len(set(free)):
                raise AssertionError("duplicate free page")
            if any(not 0 <= b < self.num_blocks for b in free):
                raise AssertionError("invalid free page")
            live: list[BlockHandle] = []
            for request_id, record in self._owners.items():
                if len(record.blocks) != ceil(record.num_tokens / self.block_size):
                    raise AssertionError("committed block table size mismatch")
                live.extend(record.blocks)
                if record.pending:
                    pending = record.pending
                    if pending.request_id != request_id or pending.original != tuple(record.blocks):
                        raise AssertionError("reservation does not match committed state")
                    if len(pending.proposed) != ceil(pending.target_tokens / self.block_size):
                        raise AssertionError("reservation table length mismatch")
                    live.extend(pending.fresh)
            ids = [b.block_id for b in live]
            if len(ids) != len(set(ids)):
                raise AssertionError("duplicate live ownership")
            for block in live:
                if not self.is_live(block):
                    raise AssertionError("stale handle still owned")
            for idx in range(self.num_blocks):
                if self._refs[idx] != (idx in ids) or (idx in free) == (idx in ids):
                    raise AssertionError("free/refcount/ownership disagree")
            if len(free) + len(live) != self.num_blocks:
                raise AssertionError("total page count mismatch")
