from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.trends import signals
from rag_trend_radar.trends.frames import Frame
from rag_trend_radar.trends.topics import Pool

WEEK = date(2001, 6, 3)


def _unit(v: NDArray[np.float32]) -> NDArray[np.float32]:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _pool(vectors: NDArray[np.float32], days: list[date], press: list[bool]) -> Pool:
    n = len(vectors)
    ids = np.arange(n, dtype=np.int64)
    return Pool(ids, ids, ["" for _ in range(n)], vectors, days, press, "store-a")


def test_a_blobs_centre_is_its_members_average_with_length_one() -> None:
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    frame = Frame(WEEK, np.arange(3, dtype=np.int64), np.array([0, 0, -1], dtype=np.int64))
    centre = signals.centres(frame, vectors)[0]
    assert np.allclose(centre, np.array([1, 1]) / np.sqrt(2))


def test_a_stray_near_one_blob_counts_for_it_and_one_between_two_doesnt() -> None:
    rng = np.random.default_rng(0)
    a = np.eye(8, dtype=np.float32)[0] + rng.normal(0, 0.02, (30, 8))
    b = np.eye(8, dtype=np.float32)[1] + rng.normal(0, 0.02, (30, 8))
    near = np.eye(8, dtype=np.float32)[0][None]
    between = (np.eye(8, dtype=np.float32)[0] + np.eye(8, dtype=np.float32)[1])[None]
    far = np.eye(8, dtype=np.float32)[5][None]
    vectors = _unit(np.vstack([a, b, near, between, far]).astype(np.float32))
    labels = np.array([0] * 30 + [1] * 30 + [-1] * 3, dtype=np.int64)
    counted = signals.counted(Frame(WEEK, np.arange(63, dtype=np.int64), labels), vectors)
    assert counted[60:].tolist() == [0, -1, -1]
    assert counted[:60].tolist() == labels[:60].tolist()


def test_counting_a_frame_with_no_blobs_leaves_it_alone() -> None:
    labels = np.array([-1, -1], dtype=np.int64)
    frame = Frame(WEEK, np.arange(2, dtype=np.int64), labels)
    assert signals.counted(frame, np.eye(2, dtype=np.float32)).tolist() == [-1, -1]


def test_a_blob_growing_with_all_mail_isnt_surging_but_one_outgrowing_it_is() -> None:
    steady = signals.surge(recent=20, before=95, all_recent=200, all_before=1000)
    assert abs(steady) < 1
    assert signals.surge(recent=120, before=95, all_recent=200, all_before=1000) > 90


def test_when_all_mail_doubles_a_steady_blobs_surge_stays_near_nothing() -> None:
    quiet = signals.surge(recent=10, before=95, all_recent=100, all_before=1000)
    busy = signals.surge(recent=20, before=95, all_recent=200, all_before=1000)
    assert abs(quiet) < 1
    assert abs(busy) < 1


def test_recent_mail_and_press_are_counted_apart_from_older_mail() -> None:
    vectors = np.tile(np.eye(4, dtype=np.float32)[0], (6, 1))
    days = [WEEK - timedelta(days=d) for d in (1, 2, 27, 28, 60, 90)]
    press = [True, False, False, True, False, False]
    labels = np.array([0] * 6, dtype=np.int64)
    frame = Frame(WEEK, np.arange(6, dtype=np.int64), labels)
    found = signals.counts(frame, _pool(vectors, days, press), labels)[0]
    assert (found.chunks, found.mail, found.recent, found.recent_press) == (6, 6, 3, 1)


def test_a_steady_blob_isnt_moving_and_one_changing_subject_is() -> None:
    a, b = np.eye(4, dtype=np.float32)[:2]
    weeks = list(range(10))
    assert signals.moving(weeks, [a] * 10)[-1] < 1e-6
    turning = signals.moving(weeks, [a] * 5 + [b] * 5)
    assert turning[4] < 1e-6 < turning[-1]


def test_movement_fades_once_a_blob_settles() -> None:
    a, b = np.eye(4, dtype=np.float32)[:2]
    found = signals.moving(list(range(60)), [a] * 10 + [b] * 50)
    assert found[-1] < max(found) / 10
