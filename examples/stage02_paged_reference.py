"""Stage 02 single-request physical paging and the dense oracle (random toy weights)."""

import torch

from nano_deepinfer.cache.paged_kv import PagedKVStore
from nano_deepinfer.dense_gpt2 import DenseGPT2Config, ReferenceGPT2
from nano_deepinfer.engine import LLMEngine
from nano_deepinfer.foundation import SamplingParams


def main() -> None:
    torch.manual_seed(2026)
    config = DenseGPT2Config(41, 16, 4, 2, 32)
    model = ReferenceGPT2(config).eval()
    pages = PagedKVStore(16, 4, config.num_layers, config.num_heads,
                         config.hidden_size // config.num_heads)
    params = SamplingParams(max_new_tokens=6, seed=1)
    reference = LLMEngine(model).generate((1, 3, 5), params)
    paged = LLMEngine(model, paged_cache=pages).generate((1, 3, 5), params)
    assert reference.generated_token_ids == paged.generated_token_ids
    pages.assert_consistent()
    assert pages.manager.stats().used == 0
    print("Token-exact Dense vs Paged KV PASS:", paged.generated_token_ids)
    print("Reserved physical KV bytes:", pages.reserved_bytes)
    print("Attention still materializes dense KV. No speedup is claimed.")


if __name__ == "__main__":
    main()
