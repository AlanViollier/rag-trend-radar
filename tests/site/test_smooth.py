import numpy as np

from rag_trend_radar.site import smooth


def test_a_missing_week_is_drawn_straight() -> None:
    assert smooth.fill([2, 4], [10, 20]).tolist() == [10, 15, 20]


def test_one_odd_week_disappears_under_the_median() -> None:
    values = np.array([10.0, 10.0, 90.0, 10.0, 10.0])
    assert smooth.median(values, 2).tolist() == [10.0] * 5


def test_the_mean_uses_only_the_weeks_that_exist_at_the_edges() -> None:
    assert smooth.mean(np.array([0.0, 6.0, 0.0]), 1).tolist() == [3.0, 2.0, 3.0]
