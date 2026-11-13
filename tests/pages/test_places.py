import numpy as np

from rag_trend_radar.pages.radar import places


def test_two_overlapping_bubbles_are_pushed_apart() -> None:
    want = np.array([[[0.5, 0.5]], [[0.51, 0.5]]])
    mail = np.full((2, 1), 2500.0)
    placed = places.pack(want, mail)
    gap = float(np.linalg.norm(placed[0, 0] - placed[1, 0]))
    assert gap > float(places.radius(mail[0])[0]) * 2


def test_a_dead_week_stays_dead() -> None:
    want = np.array([[[0.5, 0.5], [np.nan, np.nan]]])
    placed = places.pack(want, np.array([[100.0, 0.0]]))
    assert not np.isnan(placed[0, 0]).any()
    assert np.isnan(placed[0, 1]).all()


def test_a_cluster_drifts_towards_a_new_place_instead_of_jumping() -> None:
    want = np.array([[[0.2, 0.2], [0.8, 0.8]]])
    placed = places.pack(want, np.full((1, 2), 100.0))
    assert abs(placed[0, 1, 0] - placed[0, 0, 0]) < 0.1
