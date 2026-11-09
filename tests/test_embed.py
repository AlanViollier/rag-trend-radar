import hashlib

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar import embed as embedding
from rag_trend_radar.exceptions import EmbeddingFailed

DIMS = 8


class _Model:
    """Stands in for a sentence-transformers model: a fixed vector per text, or garbage."""

    def __init__(self, broken_past: int) -> None:
        self.broken_past = broken_past
        self.calls: list[int] = []

    def encode_document(self, texts: list[str], **_: object) -> NDArray[np.float32]:
        self.calls.append(len(texts))
        if len(texts) > self.broken_past or "poison" in texts:
            # Out of memory: normalised, finite, and wrong.
            rng = np.random.default_rng(len(self.calls))
            return _unit(rng.normal(size=(len(texts), DIMS)))
        return np.stack([_vector(t) for t in texts])

    encode_query = encode_document


def _vector(text: str) -> NDArray[np.float32]:
    seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest())
    return np.asarray(_unit(np.random.default_rng(seed).normal(size=(1, DIMS)))[0], np.float32)


def _unit(v: NDArray[np.float64]) -> NDArray[np.float32]:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def model(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _Model:
    fake = _Model(broken_past=request.param)
    monkeypatch.setattr(embedding, "_load", lambda _name: fake)
    monkeypatch.setattr(embedding, "_free_gpu", lambda: None)
    embedding._sentinel.cache_clear()
    return fake


@pytest.mark.parametrize("model", [100], indirect=True)
def test_every_gpu_batch_carries_the_sentinel_and_returns_only_the_texts(model: _Model) -> None:
    texts = [f"email {n}" for n in range(70)]
    vectors = embedding.embed(texts, "test/fake", DIMS)
    assert np.allclose(vectors, np.stack([_vector(t) for t in texts]))
    # 31 + 31 + 8 texts, each with the sentinel; after the first, the sentinel alone once.
    assert model.calls == [32, 1, 32, 9]


@pytest.mark.parametrize("model", [16], indirect=True)
def test_a_batch_that_runs_out_of_memory_is_split_until_it_fits(model: _Model) -> None:
    texts = [f"email {n}" for n in range(40)]
    vectors = embedding.embed(texts, "test/fake", DIMS)
    assert np.allclose(vectors, np.stack([_vector(t) for t in texts]))


@pytest.mark.parametrize("model", [100], indirect=True)
def test_a_text_that_fails_alone_is_skipped_or_raises(model: _Model) -> None:
    texts = ["email 1", "poison", "email 2"]
    vectors = embedding.embed(texts, "test/fake", DIMS, skip_failed=True)
    assert np.isnan(vectors[1]).all()
    assert np.allclose(vectors[[0, 2]], np.stack([_vector("email 1"), _vector("email 2")]))
    with pytest.raises(EmbeddingFailed, match="even alone"):
        embedding.embed(texts, "test/fake", DIMS)


@pytest.mark.parametrize("model", [0], indirect=True)
def test_a_model_broken_for_every_text_stops_even_a_long_run(model: _Model) -> None:
    with pytest.raises(EmbeddingFailed, match="every vector"):
        embedding.embed(["email 1"], "test/fake", DIMS, skip_failed=True)


@pytest.mark.parametrize("model", [100], indirect=True)
def test_nothing_to_embed_returns_no_rows(model: _Model) -> None:
    assert embedding.embed([], "test/fake", DIMS).shape == (0, DIMS)


@pytest.mark.parametrize("model", [10_000], indirect=True)
def test_a_call_gives_the_gpus_memory_back(model: _Model, monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[bool] = []
    monkeypatch.setattr(embedding, "_free_gpu", lambda: freed.append(True))
    embedding.embed(["a", "b"], "test/fake", DIMS)
    assert freed == [True]
