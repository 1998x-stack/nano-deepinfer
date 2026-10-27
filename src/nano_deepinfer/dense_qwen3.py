# mypy: ignore-errors
"""Standalone Qwen3 Dense eager reference: RoPE, GQA, Q/K RMSNorm, SwiGLU.

No Hugging Face forward pass or vLLM runtime is used here. CPU-first correctness
implementation with an unpaged per-layer KV cache. Batch size is strictly one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

QwenKV = tuple[tuple[Tensor, Tensor], ...]


@dataclass(frozen=True, slots=True)
class DenseQwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_positions: int
    rope_theta: float = 1_000_000.0
    rms_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        ints = (
            self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers,
            self.num_heads, self.num_kv_heads, self.head_dim, self.max_positions,
        )
        if any(value <= 0 for value in ints):
            raise ValueError("all Qwen3 dimensions must be positive")
        if self.head_dim % 2 or self.num_heads % self.num_kv_heads:
            raise ValueError("RoPE head dimension must be even; KV heads must divide Q heads")
        if self.rope_theta <= 1 or not 0 < self.rms_eps < 1:
            raise ValueError("invalid RoPE or RMSNorm configuration")


class QwenRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * y.to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class QwenAttention(nn.Module):
    def __init__(self, cfg: DenseQwen3Config) -> None:
        super().__init__()
        self.nh, self.nkv, self.dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = QwenRMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = QwenRMSNorm(cfg.head_dim, cfg.rms_eps)

    def _apply_rope(self, x: Tensor, pos: Tensor) -> Tensor:
        freq = 1 / (self.theta ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device) / self.dim
        ))
        angles = torch.outer(pos.float(), freq)
        emb = torch.cat((angles, angles), dim=-1)
        cos = emb.cos().to(x.dtype)[None, None, :, :]
        sin = emb.sin().to(x.dtype)[None, None, :, :]
        return x * cos + _rotate_half(x) * sin

    def forward(
        self, x: Tensor, pos: Tensor, previous: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, n, _ = x.shape
        def reshape(y: Tensor, heads: int) -> Tensor:
            return y.view(batch, n, heads, self.dim).transpose(1, 2)

        q = self.q_norm(reshape(self.q_proj(x), self.nh))
        k = self.k_norm(reshape(self.k_proj(x), self.nkv))
        v = reshape(self.v_proj(x), self.nkv)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)
        prefix = 0
        if previous is not None:
            pk, pv = previous
            prefix = pk.shape[-2]
            if pk.shape != pv.shape or pk.shape != (1, self.nkv, prefix, self.dim):
                raise ValueError("invalid Qwen3 layer cache")
            if pk.device != k.device or pk.dtype != k.dtype:
                raise ValueError("Qwen3 layer cache device/dtype mismatch")
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)
        # Keep compressed KV in cache; expand only for attention computation.
        cache = (k, v)
        expand = self.nh // self.nkv
        kr = k.repeat_interleave(expand, dim=1)
        vr = v.repeat_interleave(expand, dim=1)
        logits = (q @ kr.transpose(-2, -1)) * (self.dim ** -0.5)
        key_pos = torch.arange(prefix + n, device=x.device)[None, :]
        mask = key_pos <= pos[:, None]
        logits = logits.masked_fill(~mask[None, None, :, :], torch.finfo(logits.dtype).min)
        attn = F.softmax(logits.float(), dim=-1).to(q.dtype)
        result = attn @ vr
        return self.o_proj(result.transpose(1, 2).contiguous().view(batch, n, -1)), cache


class QwenMLP(nn.Module):
    def __init__(self, cfg: DenseQwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, cfg: DenseQwen3Config) -> None:
        super().__init__()
        self.self_attn = QwenAttention(cfg)
        self.mlp = QwenMLP(cfg)
        self.input_layernorm = QwenRMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.post_attention_layernorm = QwenRMSNorm(cfg.hidden_size, cfg.rms_eps)

    def forward(
        self, x: Tensor, pos: Tensor, previous: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        residual, cache = self.self_attn(self.input_layernorm(x), pos, previous)
        x = x + residual
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class QwenBody(nn.Module):
    def __init__(self, cfg: DenseQwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(QwenBlock(cfg) for _ in range(cfg.num_layers))
        self.norm = QwenRMSNorm(cfg.hidden_size, cfg.rms_eps)


class ReferenceQwen3(nn.Module):
    """Tiny/actual HF Qwen3 Dense weights, but completely independent forward."""

    def __init__(self, config: DenseQwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = QwenBody(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor, cache: QwenKV | None = None) -> tuple[Tensor, QwenKV]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Stage 01 Qwen3 only supports nonempty batch of one")
        if input_ids.dtype != torch.long:
            raise TypeError("token IDs must be torch.long")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("token ID outside vocabulary")
        if cache is not None and len(cache) != self.config.num_layers:
            raise ValueError("incomplete layer cache")
        offset = 0 if cache is None else cache[0][0].shape[-2]
        if cache is not None and any(
            key.shape[-2] != offset or value.shape[-2] != offset for key, value in cache
        ):
            raise ValueError("inconsistent cache position")
        length = input_ids.shape[1]
        if length + offset > self.config.max_positions:
            raise ValueError("context exceeds maximum positional embeddings")
        pos = torch.arange(offset, offset + length, device=input_ids.device)
        x = self.model.embed_tokens(input_ids)
        updated = []
        for idx, layer in enumerate(self.model.layers):
            x, kv = layer(x, pos, None if cache is None else cache[idx])
            updated.append(kv)
        return self.lm_head(self.model.norm(x)), tuple(updated)

    @classmethod
    def from_hf_qwen3(cls, hf: Any) -> ReferenceQwen3:
        config = hf.config
        if config.model_type != "qwen3":
            raise ValueError("model must be a Qwen3 Dense causal LM")
        if getattr(config, "rope_scaling", None) not in (None, {}):
            raise ValueError("RoPE scaling variants are not supported yet")
        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError("requires SwiGLU/SiLU activation")
        if bool(getattr(config, "use_sliding_window", False)):
            raise ValueError("sliding-window Qwen3 variants are not supported")
        ours = cls(DenseQwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
            max_positions=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1_000_000.0),
            rms_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            tie_word_embeddings=config.tie_word_embeddings,
        ))
        ref = hf.state_dict()
        target = ours.state_dict()
        if ref.keys() != target.keys():
            missing, unknown = set(target) - set(ref), set(ref) - set(target)
            raise ValueError(f"weight schema mismatch: missing={sorted(missing)}, unexpected={sorted(unknown)}")
        for key, tensor in target.items():
            if ref[key].shape != tensor.shape:
                raise ValueError(f"weight dimension mismatch: {key}")
        ours.load_state_dict(ref, strict=True)
        return ours.eval()
