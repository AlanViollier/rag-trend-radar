from datetime import date

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.trends import assign, topics
from rag_trend_radar.trends.words import keywords, reading_text


def _unit(v: NDArray[np.float32]) -> NDArray[np.float32]:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _blobs() -> tuple[NDArray[np.float32], list[str]]:
    rng = np.random.default_rng(0)
    centres = np.eye(16, dtype=np.float32)[:3]
    tight = [centres[n] + rng.normal(0, 0.01, (40, 16)) for n in range(3)]
    stray = rng.normal(0, 1, (10, 16))
    texts = [
        *(f"Subject: Gas {i}\nFrom: a@b.c\n\ngas pipeline thanks" for i in range(40)),
        *(f"Subject: Power {i}\nFrom: a@b.c\n\nblackout grid thanks" for i in range(40)),
        *(f"Subject: Football {i}\nFrom: a@b.c\n\nfantasy league thanks" for i in range(40)),
        *("Subject: Stray\nFrom: a@b.c\n\nsomething else" for _ in range(10)),
    ]
    return _unit(np.vstack([*tight, stray]).astype(np.float32)), texts


def test_clusters_are_found_without_being_told_how_many() -> None:
    vectors, _ = _blobs()
    labels = topics.cluster(vectors, min_size=10, min_samples=3)
    assert len(set(labels.tolist()) - {topics.NOISE}) == 3
    assert [len(set(labels[i : i + 40].tolist())) for i in (0, 40, 80)] == [1, 1, 1]
    assert len({labels[0], labels[40], labels[80]}) == 3


def test_each_topic_is_described_by_its_own_words_and_mail() -> None:
    vectors, texts = _blobs()
    vectors, texts = vectors[:120], texts[:120]
    chosen = topics.Pool(
        np.arange(len(texts), dtype=np.int64),
        np.arange(len(texts), dtype=np.int64),
        texts,
        vectors,
        [date(2000, 1, 1)] * len(texts),
        [False] * len(texts),
    )
    found = topics.describe(chosen, vectors, topics.cluster(vectors, 10, 3))
    assert {f.chunks for f in found} == {40}
    by_word = {word: f for f in found for word in f.keywords[:3]}
    assert "pipeline" in by_word
    assert "blackout" in by_word
    assert "fantasy" in by_word
    assert "pipeline" not in by_word["blackout"].keywords
    assert by_word["blackout"].nearest[0].startswith("Power")
    assert all(0.9 < f.cohesion <= 1.0 for f in found)


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
    vectors, _ = _blobs()
    first = topics.squash(vectors, components=3, neighbors=10, seed=1)
    assert first.shape == (len(vectors), 3)
    assert np.array_equal(first, topics.squash(vectors, components=3, neighbors=10, seed=1))


def test_shortened_vectors_are_length_one_again() -> None:
    vectors = _unit(np.random.default_rng(0).normal(0, 1, (5, 16)).astype(np.float32))
    short = topics.shorten(vectors, 4)
    assert short.shape == (5, 4)
    assert np.allclose(np.linalg.norm(short, axis=1), 1.0)


def test_a_cluster_of_one_long_document_is_not_a_topic() -> None:
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    documents = np.array([7] * 30 + list(range(30)), dtype=np.int64)
    kept = topics.drop_thin(labels, documents)
    assert (kept[:30] == topics.NOISE).all()
    assert (kept[30:] == 1).all()


def test_a_chunk_goes_to_its_nearest_centre_with_how_close_it_is() -> None:
    vectors = np.eye(3, dtype=np.float32)
    labels = np.array([5, 9, assign.UNASSIGNED], dtype=np.int64)
    numbers, centres = assign.centres_of(vectors, labels)
    assert numbers.tolist() == [5, 9]
    rows, best = assign.nearest(vectors, centres)
    assert rows[:2].tolist() == [0, 1]
    assert best[0] == 1.0
    assert best[2] == 0.0


def test_a_cluster_of_unrelated_mail_is_too_loose_to_be_a_topic() -> None:
    tight, _ = _blobs()
    loose = _unit(np.random.default_rng(1).normal(0, 1, (40, 16)).astype(np.float32))
    vectors = np.vstack([tight[:40], loose])
    labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    kept, dropped = topics.drop_loose(labels, vectors)
    assert dropped == 1
    assert (kept[:40] == 0).all()
    assert (kept[40:] == topics.NOISE).all()
