# mypy: ignore-errors
"""Independent GPT-2-style *reference* decoder, for correctness research only.

No vLLM, HF model forward, GPU-specific kernels, batching or paged KV are used.
Torch is an optional Stage 01 dependency. Only one request/batch element is supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DenseGPT2Config:
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    max_positions: int
    intermediate_size: int | None = None
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_heads", "num_layers", "max_positions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        if not 0 < self.norm_eps < 1:
            raise ValueError("norm_eps must be in (0, 1)")


LayerKV = tuple[Tensor, Tensor]
ModelKV = tuple[LayerKV, ...]


class ReferenceSelfAttention(nn.Module):
    def __init__(self, cfg: DenseGPT2Config) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.c_attn = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: Tensor, past: LayerKV | None) -> tuple[Tensor, LayerKV]:
        batch_size, t, dim = x.shape
        q, k, v = self.c_attn(x).split(dim, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        past_len = 0
        if past is not None:
            if past[0].shape != past[1].shape or past[0].shape[:2] != (1, self.n_heads):
                raise ValueError("invalid KV head shape")
            past_len = past[0].shape[-2]
            k, v = torch.cat([past[0], k], dim=-2), torch.cat([past[1], v], dim=-2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(past_len, past_len + t, device=x.device).unsqueeze(-1)
        keys = torch.arange(past_len + t, device=x.device).unsqueeze(0)
        allowed = keys <= positions
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, t, dim)
        return self.c_proj(attended), (k, v)


class ReferenceMLP(nn.Module):
    def __init__(self, cfg: DenseGPT2Config) -> None:
        super().__init__()
        inner = cfg.intermediate_size or 4 * cfg.hidden_size
        self.c_fc = nn.Linear(cfg.hidden_size, inner)
        self.c_proj = nn.Linear(inner, cfg.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class ReferenceBlock(nn.Module):
    def __init__(self, cfg: DenseGPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = ReferenceSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = ReferenceMLP(cfg)

    def forward(self, x: Tensor, cache: LayerKV | None) -> tuple[Tensor, LayerKV]:
        attention, new_cache = self.attn(self.ln_1(x), cache)
        x = x + attention
        return x + self.mlp(self.ln_2(x)), new_cache


class ReferenceTransformer(nn.Module):
    def __init__(self, cfg: DenseGPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(cfg.max_positions, cfg.hidden_size)
        self.h = nn.ModuleList(ReferenceBlock(cfg) for _ in range(cfg.num_layers))
        self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)


class ReferenceGPT2(nn.Module):
    """Small dense causal LM with explicit full-prefix and incremental forward paths."""

    def __init__(self, cfg: DenseGPT2Config) -> None:
        super().__init__()
        self.config = cfg
        self.transformer = ReferenceTransformer(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: Tensor, cache: ModelKV | None = None) -> tuple[Tensor, ModelKV]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Stage 01 requires nonempty [1, tokens] inputs")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input token outside vocabulary")
        if cache is not None and len(cache) != self.config.num_layers:
            raise ValueError("cache layer count mismatch")
        offset = cache[0][0].shape[-2] if cache else 0
        if cache is not None and any(k.shape[-2] != offset or v.shape[-2] != offset for k, v in cache):
            raise ValueError("cache position mismatch")
        t = input_ids.shape[1]
        if offset + t > self.config.max_positions:
            raise ValueError("context length exceeds max_positions")
        positions = torch.arange(offset, offset + t, device=input_ids.device)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(positions)[None, :, :]
        next_caches: list[LayerKV] = []
        for layer_idx, block in enumerate(self.transformer.h):
            x, new_cache = block(x, None if cache is None else cache[layer_idx])
            next_caches.append(new_cache)
        logits = self.lm_head(self.transformer.ln_f(x))
        return logits, tuple(next_caches)

    @classmethod
    def from_hf_gpt2(cls, hf_model: Any) -> "ReferenceGPT2":
        """Import GPT-2 **weights**, never HF forward computation into nano runtime."""
        cfg = hf_model.config
        if cfg.model_type != "gpt2":
            raise ValueError("Stage 01 reference adapter supports GPT-2 only")
        if cfg.add_cross_attention or cfg.scale_attn_by_inverse_layer_idx or cfg.reorder_and_upcast_attn:
            raise ValueError("Unsupported GPT-2 variant features")
        if cfg.activation_function != "gelu_new":
            raise ValueError("Stage 01 GPT-2 supports gelu_new activation")
        config = DenseGPT2Config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.n_embd,
            num_heads=cfg.n_head,
            num_layers=cfg.n_layer,
            max_positions=cfg.n_positions,
            intermediate_size=cfg.n_inner,
            norm_eps=cfg.layer_norm_epsilon,
        )
        model = cls(config)
        ref = hf_model.state_dict()
        ours = model.state_dict()
        transposed_suffixes = (
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
        )
        for name in ours:
            if name not in ref:
                raise ValueError(f"missing model weight {name}")
            tensor = ref[name]
            if name.endswith(transposed_suffixes):
                tensor = tensor.transpose(0, 1).contiguous()
            if tensor.shape != ours[name].shape:
                raise ValueError(f"weight shape mismatch for {name}")
            ours[name] = tensor.detach().clone()
        model.load_state_dict(ours, strict=True)
        return model.eval()
