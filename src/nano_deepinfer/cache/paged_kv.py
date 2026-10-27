# mypy: ignore-errors
"""Stage 02 contiguous-attention-compatible *paged physical KV storage*.

Model attention remains the Stage 01 dense reference kernel. Reading cached
pages materializes dense tensors and is not a PagedAttention speedup.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor

from nano_deepinfer.cache.block_manager import (
    BlockManager,
    BlockTable,
    KVStateError,
)

DenseKV: TypeAlias = tuple[tuple[Tensor, Tensor], ...]


class PagedKVStore:
    """Fixed-size physical pages with synchronous transactional writes.

    Page shape is (kv_heads, block_size, head_dim) for each K and V per layer.
    No cache sharing between requests and no asynchronous readers. A failed
    write never publishes an incomplete logical block table.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_layers < 1 or kv_heads < 1 or head_dim < 1:
            raise ValueError("invalid KV tensor dimensions")
        self.manager = BlockManager(num_blocks, block_size)
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        page_shape = (num_blocks, kv_heads, block_size, head_dim)
        self._keys = [torch.empty(page_shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._values = [torch.empty(page_shape, dtype=dtype, device=device) for _ in range(num_layers)]

    def _check(self, dense: DenseKV) -> int:
        if len(dense) != self.num_layers:
            raise ValueError("wrong number of cache layers")
        length = None
        for k, v in dense:
            if k.ndim != 4 or v.shape != k.shape or k.shape[0] != 1:
                raise ValueError("cache tensors must be matching [1,H,T,D]")
            if k.shape[1] != self.kv_heads or k.shape[3] != self.head_dim:
                raise ValueError("cache head geometry mismatch")
            if k.dtype != self.dtype or v.dtype != self.dtype:
                raise ValueError("cache dtype mismatch")
            if k.device != self.device or v.device != self.device:
                raise ValueError("cache device mismatch")
            if length is not None and k.shape[2] != length:
                raise ValueError("KV layers disagree on token length")
            length = k.shape[2]
        if length is None or length < 1:
            raise ValueError("empty cache cannot be published")
        return length

    def write(self, request_id: str, dense: DenseKV) -> BlockTable:
        """Publish full current KV (prefill or extended decode) transactionally."""
        target = self._check(dense)
        created = False
        try:
            try:
                original = self.manager.table(request_id)
            except KVStateError:
                self.manager.create(request_id)
                created = True
                original = self.manager.table(request_id)
            if target < original.num_tokens:
                raise ValueError("cache length cannot regress")
            reservation = self.manager.begin(request_id, target)
        except Exception:
            if created:
                self.manager.release(request_id)
            raise
        try:
            new_handles = set(reservation.fresh)
            for page_idx, handle in enumerate(reservation.proposed):
                if handle not in new_handles:
                    continue
                start = page_idx * self.manager.block_size
                end = min(start + self.manager.block_size, target)
                for layer, (k, v) in enumerate(dense):
                    count = end - start
                    # Copy all occupied positions into fresh pages; partially
                    # filled old last pages are always replaced by COW.
                    self._keys[layer][handle.block_id, :, :count, :].copy_(
                        k[0, :, start:end, :]
                    )
                    self._values[layer][handle.block_id, :, :count, :].copy_(
                        v[0, :, start:end, :]
                    )
        except BaseException:
            self.manager.abort(reservation)
            if created:
                self.manager.release(request_id)
            raise
        return self.manager.commit(reservation)

    def read(self, request_id: str) -> DenseKV:
        table = self.manager.table(request_id)
        if table.num_tokens == 0:
            raise KVStateError("empty KV cache")
        for block in table.blocks:
            if not self.manager.is_live(block):
                raise KVStateError("stale page handle")
        result: list[tuple[Tensor, Tensor]] = []
        for layer in range(self.num_layers):
            key_parts = []
            value_parts = []
            for block in table.blocks:
                key_parts.append(self._keys[layer][block.block_id])
                value_parts.append(self._values[layer][block.block_id])
            key = torch.cat(key_parts, dim=1)[:, :table.num_tokens, :].unsqueeze(0)
            val = torch.cat(value_parts, dim=1)[:, :table.num_tokens, :].unsqueeze(0)
            result.append((key, val))
        return tuple(result)

    def release(self, request_id: str) -> bool:
        return self.manager.release(request_id)

    def assert_consistent(self) -> None:
        self.manager.assert_consistent()

    @property
    def reserved_bytes(self) -> int:
        """Physical page storage only; excludes temporary dense materialization."""
        element_bytes = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.manager.num_blocks * self.manager.block_size * self.num_layers
            * self.kv_heads * self.head_dim * 2 * element_bytes
        )
