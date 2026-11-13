from datetime import date, timedelta

import numpy as np

from rag_trend_radar.radar import links
from rag_trend_radar.radar.frames import Frame

START = date(2001, 1, 7)


def _frame(week: int, labels: list[int]) -> Frame:
    return Frame(
        START + timedelta(weeks=week),
        np.arange(len(labels), dtype=np.int64),
        np.array(labels, dtype=np.int64),
    )


def test_the_same_chunks_next_week_are_the_same_cluster() -> None:
    first = _frame(0, [0] * 20 + [1] * 20)
    second = _frame(1, [7] * 20 + [3] * 20)
    assert links.same(first, second) == {7: 0, 3: 1}


def test_a_few_shared_chunks_link_nothing() -> None:
    first = _frame(0, [0] * 9 + [-1] * 40)
    second = _frame(1, [0] * 9 + [1] * 40)
    assert links.same(first, second) == {}


def test_a_cluster_splitting_in_two_continues_as_the_bigger_piece() -> None:
    first = _frame(0, [0] * 60)
    second = _frame(1, [1] * 40 + [2] * 20)
    assert links.same(first, second) == {1: 0}


def test_a_cluster_can_miss_weeks_and_keep_its_lineage() -> None:
    frames = [
        _frame(0, [0] * 20),
        _frame(1, [-1] * 20),
        _frame(2, [-1] * 20),
        _frame(3, [4] * 20),
    ]
    found = links.lineages(frames)
    assert found[(frames[0].end, 0)] == found[(frames[3].end, 4)]


def test_a_cluster_gone_too_long_comes_back_as_a_new_one() -> None:
    frames = [_frame(0, [0] * 20), _frame(1, [-1] * 20), _frame(2, [-1] * 20), _frame(3, [4] * 20)]
    found = links.lineages(frames, bridge=1)
    assert found[(frames[0].end, 0)] != found[(frames[3].end, 4)]


def test_a_lineage_taken_by_a_live_cluster_cant_be_bridged_to_as_well() -> None:
    frames = [
        _frame(0, [0] * 40),
        _frame(1, [1] * 40),
        _frame(2, [2] * 40),
    ]
    found = links.lineages(frames)
    assert len({found[(f.end, int(f.labels[0]))] for f in frames}) == 1


def test_later_weeks_dont_change_earlier_lineages() -> None:
    frames = [_frame(0, [0] * 20 + [1] * 20), _frame(1, [0] * 20 + [1] * 20), _frame(2, [5] * 40)]
    early = links.lineages(frames[:2])
    assert {k: v for k, v in links.lineages(frames).items() if k in early} == early
