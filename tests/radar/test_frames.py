from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar.exceptions import OtherStore
from rag_trend_radar.radar import build, frames
from rag_trend_radar.radar.clustering import Pool
from rag_trend_radar.run_log import RunLog

D = date(2001, 3, 4)
SUBJECTS = ("gas pipeline", "power grid", "fantasy football")


def _world(weeks_after: int) -> Pool:
    """Three steady subjects for 30 weeks up to `D`, and more mail, with a new subject, after it."""
    rng = np.random.default_rng(0)
    rows: list[tuple[int, date]] = []
    for day in range(210):
        rows += [(s, D - timedelta(days=day)) for s in range(3) for _ in range(2)]
    for day in range(1, weeks_after * 7 + 1):
        rows += [(3, D + timedelta(days=day)) for _ in range(6)]
    rows.sort(key=lambda r: r[1])
    axes = np.eye(16, dtype=np.float32)
    vectors = np.array([axes[s] + rng.normal(0, 0.03, 16) for s, _ in rows], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    named = (*SUBJECTS, "new subject")
    texts = [
        f"Subject: {named[s]}\nFrom: a{i}@b.c\n\n{named[s]} today" for i, (s, _) in enumerate(rows)
    ]
    ids = np.arange(len(rows), dtype=np.int64)
    return Pool(ids, ids, texts, vectors, [d for _, d in rows], [False] * len(rows), "store-a")


def _rows(
    chosen: Pool, ends: list[date], folder: Path
) -> dict[tuple[int, date], tuple[object, ...]]:
    frames.make(chosen, ends, 1, folder, RunLog("test", {}), min_size=20)
    made = frames.load(chosen, ends, folder)
    return {
        (w.cluster, w.week): (w.label, w.centre.tolist(), w.words, w.counts, w.moving, w.other_seed)
        for w in build.cluster_weeks(chosen, made, made)
    }


def test_rows_up_to_a_week_dont_depend_on_mail_after_it(tmp_path: Path) -> None:
    early = [D - timedelta(weeks=1), D]
    later = [*early, D + timedelta(weeks=1)]
    known = _rows(_world(0), early, tmp_path / "known")
    replayed = _rows(_world(3), later, tmp_path / "replayed")
    assert known
    assert {k: v for k, v in replayed.items() if k[1] <= D} == known


def test_a_window_holds_26_weeks_up_to_its_end() -> None:
    chosen = _world(1)
    rows = frames.window(chosen, D)
    picked = [chosen.days[i] for i in rows]
    assert max(picked) == D
    assert min(picked) == D - timedelta(weeks=26) + timedelta(days=1)


def test_week_ends_run_from_first_to_last_a_week_apart() -> None:
    ends = frames.week_ends(date(2000, 7, 2), date(2000, 7, 23))
    assert ends == [date(2000, 7, d) for d in (2, 9, 16, 23)]
    assert frames.week_ends()[0] == frames.FIRST
    assert frames.week_ends()[-1] == frames.LAST
    assert len(frames.week_ends()) == 87


def test_a_chunk_starts_where_it_was_and_a_new_one_among_its_nearest() -> None:
    vectors: NDArray[np.float32] = np.eye(3, dtype=np.float32)[
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    ]
    old_rows = np.arange(12, dtype=np.int64)
    old_points = np.vstack([np.full((6, 2), 1.0), np.full((6, 2), 9.0)]).astype(np.float32)
    start = frames.warm_start(vectors, np.array([0, 7, 12], dtype=np.int64), old_rows, old_points)
    assert start.tolist() == [[1.0, 1.0], [9.0, 9.0], [1.0, 1.0]]


def test_a_week_already_saved_isnt_made_again(tmp_path: Path) -> None:
    chosen = _world(0)
    frames.make(chosen, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    log = RunLog("test", {})
    frames.make(chosen, [D], 1, tmp_path, log, min_size=20)
    assert log.run().stages == {}


def _without(chosen: Pool, gone: int) -> Pool:
    return chosen.take(np.flatnonzero(chosen.chunk_ids != gone))


def _labelled(made: list[frames.Frame], chosen: Pool) -> dict[int, int]:
    return dict(zip(chosen.chunk_ids[made[0].rows].tolist(), made[0].labels.tolist(), strict=True))


def test_a_deleted_chunk_drops_out_of_its_week_and_nothing_else_moves(tmp_path: Path) -> None:
    chosen = _world(0)
    frames.make(chosen, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    before = _labelled(frames.load(chosen, [D], tmp_path), chosen)
    gone = int(chosen.chunk_ids[frames.window(chosen, D)[100]])
    after = _labelled(frames.load(_without(chosen, gone), [D], tmp_path), _without(chosen, gone))
    assert gone not in after
    assert after == {k: v for k, v in before.items() if k != gone}


def test_clusters_made_after_a_delete_still_line_up_with_the_saved_chunks(tmp_path: Path) -> None:
    chosen = _world(0)
    frames.make(chosen, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    next(tmp_path.glob("*-labels-*.npy")).unlink()
    gone = int(chosen.chunk_ids[frames.window(chosen, D)[100]])
    smaller = _without(chosen, gone)
    frames.make(smaller, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    made = frames.load(smaller, [D], tmp_path)
    assert len(made[0].labels) == len(made[0].rows) == len(frames.window(chosen, D)) - 1


def test_saved_weeks_refuse_to_load_against_a_rebuilt_store(tmp_path: Path) -> None:
    chosen = _world(0)
    frames.make(chosen, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    # Same chunk ids, but a rebuild could have handed them to other mail.
    with pytest.raises(OtherStore, match="store-a, the store is now store-b"):
        frames.load(replace(chosen, store="store-b"), [D], tmp_path)
    with pytest.raises(OtherStore):
        frames.make(
            replace(chosen, store="store-b"), [D], 1, tmp_path, RunLog("test", {}), min_size=20
        )


def test_saved_weeks_with_no_store_recorded_refuse_to_load(tmp_path: Path) -> None:
    chosen = _world(0)
    frames.make(chosen, [D], 1, tmp_path, RunLog("test", {}), min_size=20)
    (tmp_path / frames.STAMP).unlink()
    with pytest.raises(OtherStore, match="unknown store"):
        frames.load(chosen, [D], tmp_path)
