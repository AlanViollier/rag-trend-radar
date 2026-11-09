import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar.trends import topics
from rag_trend_radar.trends.words import keywords, reading_text


def _unit(v: NDArray[np.float32]) -> NDArray[np.float32]:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _blobs() -> NDArray[np.float32]:
    rng = np.random.default_rng(0)
    centres = np.eye(16, dtype=np.float32)[:5]
    tight = [centres[n] + rng.normal(0, 0.01, (40, 16)) for n in range(5)]
    stray = rng.normal(0, 1, (10, 16))
    return _unit(np.vstack([*tight, stray]).astype(np.float32))


def test_clusters_are_found_without_being_told_how_many() -> None:
    labels = topics.cluster(_blobs(), min_size=10, min_samples=3)
    assert len(set(labels.tolist()) - {topics.NOISE}) == 5
    assert [len(set(labels[i : i + 40].tolist())) for i in range(0, 200, 40)] == [1] * 5


def test_a_frame_with_one_giant_blob_is_clustered_again_with_a_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps: list[int | None] = []

    def giant(points: object, min_size: int, min_samples: int, largest: int | None) -> object:
        caps.append(largest)
        return np.zeros(1000, dtype=np.int64)

    monkeypatch.setattr(topics, "_hdbscan", giant)
    topics.cluster(np.zeros((1000, 5), dtype=np.float32), min_size=20)
    assert caps == [None, 50]


def test_a_frame_without_a_giant_blob_is_clustered_once(monkeypatch: pytest.MonkeyPatch) -> None:
    caps: list[int | None] = []

    def spread(points: object, min_size: int, min_samples: int, largest: int | None) -> object:
        caps.append(largest)
        return np.arange(1000, dtype=np.int64) % 10

    monkeypatch.setattr(topics, "_hdbscan", spread)
    topics.cluster(np.zeros((1000, 5), dtype=np.float32), min_size=20)
    assert caps == [None]


def test_a_word_every_topic_uses_loses_to_one_only_this_topic_uses() -> None:
    top = keywords(["thanks thanks thanks blackout blackout", "thanks thanks thanks pipeline"])
    assert top[0][0] == "blackout"
    assert top[1][0] == "pipeline"


def test_keywords_ignore_a_lone_topic_and_common_english() -> None:
    assert keywords(["only topic here"]) == [[]]
    assert "the" not in keywords(["the blackout", "the pipeline"])[0]


def test_a_header_gives_its_subject_and_nothing_else_to_read() -> None:
    text = reading_text("Subject: Refund\nFrom: a@b.c\nDate: 2001-01-02\n\nFerc hearing.")
    assert text == "Refund Ferc hearing."


def test_the_same_seed_squashes_to_the_same_map() -> None:
    vectors = _blobs()
    first = topics.squash(vectors, components=3, neighbors=10, seed=1)
    assert first.shape == (len(vectors), 3)
    assert np.array_equal(first, topics.squash(vectors, components=3, neighbors=10, seed=1))


def test_a_map_can_start_from_where_the_last_one_ended() -> None:
    vectors = _blobs()
    first = topics.squash(vectors, components=3, neighbors=10, seed=1)
    second = topics.squash(vectors, components=3, neighbors=10, seed=1, start=first)
    cold = np.linalg.norm(topics.squash(vectors, 3, 10, seed=2) - first, axis=1).mean()
    assert np.linalg.norm(second - first, axis=1).mean() < cold


def test_a_cluster_of_one_long_document_is_not_a_topic() -> None:
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    documents = np.array([7] * 30 + list(range(30)), dtype=np.int64)
    kept = topics.drop_thin(labels, documents)
    assert (kept[:30] == topics.NOISE).all()
    assert (kept[30:] == 1).all()


def test_a_cluster_of_unrelated_mail_is_too_loose_to_be_a_topic() -> None:
    tight = _blobs()
    loose = _unit(np.random.default_rng(1).normal(0, 1, (40, 16)).astype(np.float32))
    vectors = np.vstack([tight[:40], loose])
    labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    kept = topics.drop_loose(labels, vectors)
    assert (kept[:40] == 0).all()
    assert (kept[40:] == topics.NOISE).all()
