"""Stage 02 physical KV pages and real model output equivalence."""

import pytest

torch = pytest.importorskip("torch")

from nano_deepinfer.cache.block_manager import KVCapacityError, KVStateError  # noqa: E402
from nano_deepinfer.cache.paged_kv import PagedKVStore  # noqa: E402
from nano_deepinfer.dense_gpt2 import DenseGPT2Config, ReferenceGPT2  # noqa: E402
from nano_deepinfer.dense_qwen3 import DenseQwen3Config, ReferenceQwen3  # noqa: E402
from nano_deepinfer.engine import LLMEngine  # noqa: E402
from nano_deepinfer.foundation import SamplingParams  # noqa: E402


def fake_cache(length: int):
    # Simulate true autoregressive KV: prior positions never change as length grows.
    ar = torch.arange(2 * 128 * 4, dtype=torch.float32).view(1, 2, 128, 4)[:, :, :length, :]
    return tuple((ar + 100 * i, -ar - 100 * i) for i in range(2))


def test_page_storage_round_trip_and_release() -> None:
    pool = PagedKVStore(9, 4, 2, 2, 4)
    assert pool.reserved_bytes == 9 * 4 * 2 * 2 * 4 * 2 * 4
    for length in (1, 3, 4, 5, 8, 11):
        original = fake_cache(length)
        table = pool.write("a", original)
        assert table.num_tokens == length
        for expected, actual in zip(original, pool.read("a"), strict=True):
            for left, right in zip(expected, actual, strict=True):
                assert torch.equal(left, right)
        pool.assert_consistent()
    assert pool.release("a")
    assert not pool.release("a")
    with pytest.raises(KVStateError):
        pool.read("a")
    pool.write("b", fake_cache(2))
    assert torch.equal(pool.read("b")[0][0], fake_cache(2)[0][0])
    pool.release("b")
    assert pool.manager.stats().used == 0


def test_committed_prefix_cannot_be_silently_modified() -> None:
    pool = PagedKVStore(5, 4, 2, 2, 4)
    pool.write("r", fake_cache(3))
    original = pool.manager.table("r")
    larger = list(fake_cache(5))
    k, v = larger[0]
    corrupted = k.clone()
    corrupted[0, 0, 0, 0] += 1
    larger[0] = (corrupted, v)
    with pytest.raises(ValueError, match="committed KV prefix"):
        pool.write("r", tuple(larger))
    assert pool.manager.table("r") == original
    assert pool.manager.stats().pending == 0
    pool.assert_consistent()
    pool.release("r")


def test_failed_page_write_aborts_without_changing_published_kv() -> None:
    pool = PagedKVStore(6, 4, 2, 2, 4)
    pool.write("one", fake_cache(3))
    before = pool.manager.table("one")
    first = pool.read("one")[0][0]
    original = pool._keys[0]

    class BrokenPage:
        def __getitem__(self, index: object):
            # Allow committed prefix reads; fail *after* reservation on stage-copy.
            if isinstance(index, tuple):
                raise RuntimeError("injected write fault")
            return original[index]

    pool._keys[0] = BrokenPage()
    with pytest.raises(RuntimeError, match="injected"):
        pool.write("one", fake_cache(5))
    pool._keys[0] = original
    assert pool.manager.table("one") == before
    assert torch.equal(pool.read("one")[0][0], first)
    assert pool.manager.stats().pending == 0
    pool.assert_consistent()
    pool.release("one")


@pytest.mark.parametrize("bad", ["head", "dtype", "layers", "shorter"])
def test_bad_data_rejected_before_reservation(bad: str) -> None:
    pool = PagedKVStore(4, 4, 2, 2, 4)
    pool.write("r", fake_cache(3))
    before = pool.manager.table("r")
    values = fake_cache(5)
    if bad == "head":
        values = ((values[0][0][:, :1], values[0][1][:, :1]), values[1])
    elif bad == "dtype":
        values = ((values[0][0].double(), values[0][1].double()), values[1])
    elif bad == "layers":
        values = values[:1]
    else:
        values = fake_cache(2)
    with pytest.raises(ValueError):
        pool.write("r", values)
    assert pool.manager.table("r") == before
    pool.assert_consistent()
    pool.release("r")


@pytest.mark.parametrize("name", ["gpt2", "qwen"])
@pytest.mark.parametrize("temperature", [0.0, 0.85])
def test_paged_inference_exact_token_parity_to_dense(name: str, temperature: float) -> None:
    torch.manual_seed(101)
    if name == "gpt2":
        model = ReferenceGPT2(DenseGPT2Config(41, 16, 4, 2, 24)).eval()
        pool = PagedKVStore(16, 4, 2, 4, 4)
    else:
        model = ReferenceQwen3(DenseQwen3Config(41, 32, 64, 2, 4, 2, 8, 24)).eval()
        pool = PagedKVStore(16, 4, 2, 2, 8)
    params = SamplingParams(max_new_tokens=7, temperature=temperature,
                            top_k=5, top_p=0.8, seed=23)
    ref = LLMEngine(model).generate((1, 3, 5), params, request_id="ref")
    actual = LLMEngine(model, paged_cache=pool).generate(
        (1, 3, 5), params, request_id="paged"
    )
    assert actual.generated_token_ids == ref.generated_token_ids
    assert actual.finish_reason == ref.finish_reason
    assert pool.manager.stats().free == 16
    pool.assert_consistent()


def test_paged_model_oom_releases_every_page() -> None:
    torch.manual_seed(0)
    model = ReferenceGPT2(DenseGPT2Config(41, 16, 4, 2, 24)).eval()
    with pytest.raises(ValueError, match="dimensions"):
        LLMEngine(model, paged_cache=PagedKVStore(4, 4, 2, 2, 4))
    pool = PagedKVStore(1, 4, 2, 4, 4)
    with pytest.raises(KVCapacityError):
        LLMEngine(model, paged_cache=pool).generate(
            (1, 2, 3), SamplingParams(max_new_tokens=3), request_id="oom"
        )
    assert pool.manager.stats().owners == 0
    assert pool.manager.stats().free == 1
