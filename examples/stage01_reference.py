"""Run a standalone deterministic toy Qwen3 Dense inference example.

Requires: pip install -e '.[reference]'. No model downloads; no vLLM runtime.
"""

from __future__ import annotations

import torch

from nano_deepinfer.dense_qwen3 import DenseQwen3Config, ReferenceQwen3
from nano_deepinfer.engine import LLMEngine
from nano_deepinfer.foundation import SamplingParams


def main() -> None:
    torch.manual_seed(42)
    cfg = DenseQwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=8, max_positions=32
    )
    model = ReferenceQwen3(cfg).eval()
    output = LLMEngine(model).generate((1, 2, 3), SamplingParams(max_new_tokens=4))
    print("Toy-model output token IDs:", output.generated_token_ids)
    print("This model contains random weights; output is not meaningful text.")


if __name__ == "__main__":
    main()
