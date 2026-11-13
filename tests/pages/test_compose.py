from datetime import date, timedelta
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.pages.radar import compose, places, tables
from rag_trend_radar.pages.radar.records import EVENTS, AlertNote, Event, Moment
from rag_trend_radar.radar import frames
from rag_trend_radar.radar.naming import Naming
from rag_trend_radar.records import read_records

START = date(2001, 1, 7)
ENDS = [START + timedelta(weeks=i) for i in range(6)]
WORDS = ("gas", "pipeline", "storage")


def _seen(cluster: int, week: int, *, shown: bool = True, mail: int = 100) -> tables.Seen:
    return tables.Seen(
        cluster=cluster,
        shown=shown,
        week=ENDS[week],
        centre=np.ones(4, dtype=np.float32),
        words=WORDS,
        mail=mail,
        recent=50,
        surge=10.0,
    )


def _day(week: int, documents: int = 5) -> tables.Day:
    return tables.Day(ENDS[week], documents, documents * 2, 3)


def _tables(fired: list[tables.Fired] | None = None) -> tables.Tables:
    seen = [_seen(1, 0), _seen(1, 2, mail=300), _seen(9, 1, shown=False), _seen(2, 3)]
    dust = [tables.Speck(np.ones(4, dtype=np.float32), ENDS[1])]
    return tables.Tables(seen, fired or [], [_day(0), _day(1, 7)], 100, 200, dust)


@pytest.fixture(autouse=True)
def flat_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A map without UMAP, which needs more points than these tests have."""

    def flat(rows: int) -> NDArray[np.float64]:
        return np.linspace(0.2, 0.8, 2 * rows).reshape(-1, 2)

    def spread(
        main: NDArray[np.float32], others: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return flat(len(main)), flat(len(others))

    def rise(
        main: NDArray[np.float32], start: NDArray[np.float64], others: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = flat(len(others))
        return np.column_stack([start, start[:, 0]]), np.column_stack([lower, lower[:, 0]])

    monkeypatch.setattr(places, "map_points", spread)
    monkeypatch.setattr(places, "lift", rise)


def test_a_gap_is_filled_and_a_cluster_runs_first_to_last_week() -> None:
    cluster = compose.page(_tables(), [], [], ENDS).clusters[0]
    assert (cluster.id, cluster.first, len(cluster.mail), len(cluster.x)) == (1, 0, 3, 3)


def test_a_cluster_has_a_flat_and_a_lifted_place_every_week() -> None:
    cluster = compose.page(_tables(), [], [], ENDS).clusters[0]
    assert len(cluster.x) == len(cluster.lifted_z) == 3


def test_a_moment_fills_in_its_numbers_and_makes_its_alert_bold() -> None:
    fired = tables.Fired(ENDS[3], 1, 1234.0, 2.4, ENDS[0])
    moment = Moment(day=ENDS[3], cluster=1, text="{observed} against {normal}, {mail} that week")
    data = compose.page(_tables([fired]), [], [moment], ENDS)
    assert (data.moments[0].week, data.moments[0].text) == (3, "1,234 against 2, 0 that week")
    assert (data.alerts[0].note, data.alerts[0].weight) == (data.moments[0].text, "bold")


def test_a_moment_says_how_long_after_its_event() -> None:
    fired = tables.Fired(ENDS[3], 1, 10.0, 2.0, ENDS[0])
    event = Event(id="x", day=ENDS[3] - timedelta(days=12), title="Something")
    moment = Moment(day=ENDS[3], cluster=1, text="t", event="x")
    data = compose.page(_tables([fired]), [event], [moment], ENDS)
    assert data.alerts[0].follows == "12 days after: Something"


def test_a_moment_naming_an_unknown_event_stops_the_export() -> None:
    fired = tables.Fired(ENDS[3], 1, 10.0, 2.0, ENDS[0])
    with pytest.raises(ExportRefused):
        compose.page(
            _tables([fired]), [], [Moment(day=ENDS[3], cluster=1, text="t", event="x")], ENDS
        )


def _named(kind: Literal["topic", "newsletter"] = "topic", first: date = ENDS[0]) -> Naming:
    names = ((first, "Gas"), (ENDS[2], "Pipes"))
    return Naming(cluster=1, first=first, kind=kind, family="Trading", names=names)


def test_a_read_cluster_takes_its_names_and_family() -> None:
    cluster = compose.page(_tables(), [], [], ENDS, [_named()]).clusters[0]
    assert (cluster.names, cluster.family) == (((0, "Gas"), (2, "Pipes")), "Trading")


def test_a_cluster_nobody_read_keeps_its_rough_name() -> None:
    clusters = {b.id: b for b in compose.page(_tables(), [], [], ENDS, [_named()]).clusters}
    assert (clusters[2].family, len(clusters[2].names)) == ("", 1)


def test_a_cluster_read_as_a_newsletter_leaves_the_map() -> None:
    clusters = {
        b.id: b for b in compose.page(_tables(), [], [], ENDS, [_named("newsletter")]).clusters
    }
    assert (clusters[1].shown, clusters[2].shown) == (False, True)


def test_a_name_for_the_wrong_first_week_stops_the_export() -> None:
    with pytest.raises(ExportRefused):
        compose.page(_tables(), [], [], ENDS, [_named(first=ENDS[1])])


def test_a_read_alert_takes_its_weight_and_note() -> None:
    fired = tables.Fired(ENDS[3], 1, 80.0, 2.0, ENDS[0])
    note = AlertNote(day=ENDS[3], cluster=1, weight="muted", headline="Gift ads", note="Junk.")
    alert = compose.page(_tables([fired]), [], [], ENDS, read_alerts=[note]).alerts[0]
    assert (alert.weight, alert.headline, alert.note) == ("muted", "Gift ads", "Junk.")


def test_a_note_on_no_alert_stops_the_export() -> None:
    note = AlertNote(day=ENDS[3], cluster=1, weight="bold", note="x")
    with pytest.raises(ExportRefused):
        compose.page(_tables(), [], [], ENDS, read_alerts=[note])


def test_dust_arrives_the_week_it_was_sent() -> None:
    dust = compose.page(_tables(), [], [], ENDS).dust
    assert (dust.first, len(dust.x), len(dust.lifted_z)) == ((1,), 1, 1)


def test_a_moment_with_no_alert_stops_the_export() -> None:
    with pytest.raises(ExportRefused):
        compose.page(_tables(), [], [Moment(day=ENDS[3], cluster=1, text="x")], ENDS)


def test_a_cluster_not_on_the_main_map_comes_last_and_still_has_a_place() -> None:
    last = compose.page(_tables(), [], [], ENDS).clusters[-1]
    assert (last.id, last.shown, len(last.mail), len(last.lifted_z)) == (9, False, 1, 1)


def test_the_store_grows_from_what_came_before() -> None:
    weeks = compose.page(_tables(), [], [], ENDS).weeks
    assert [w.documents for w in weeks[:3]] == [105, 112, 112]
    assert [w.chunks for w in weeks[:2]] == [210, 224]


def test_a_week_counts_only_its_own_mail() -> None:
    weeks = compose.page(_tables(), [], [], ENDS).weeks
    assert [w.mail for w in weeks[:3]] == [5, 7, 0]


def test_live_clusters_counts_shown_clusters_only() -> None:
    assert [w.live_clusters for w in compose.page(_tables(), [], [], ENDS).weeks] == [
        1,
        1,
        1,
        1,
        0,
        0,
    ]


def test_an_alert_on_a_young_cluster_is_new_on_an_older_one_speeding() -> None:
    young = tables.Fired(ENDS[3], 1, 80.4, 2.0, ENDS[0])
    older = tables.Fired(ENDS[4], 1, 90.0, 10.0, ENDS[0])
    alerts = compose.page(_tables([young, older]), [], [], ENDS).alerts
    assert [(a.week, a.kind, a.observed) for a in alerts] == [(3, "new", 80), (4, "speeding", 90)]


def test_small_floats_round_to_short_numbers() -> None:
    assert compose._rounded(np.array([0.123], dtype=np.float32), 3) == (0.123,)


def test_events_pass_through() -> None:
    event = Event(id="x", day=ENDS[0], title="Something")
    assert compose.page(_tables(), [event], [], ENDS).events == (event,)


def test_every_real_event_sits_inside_the_radars_run() -> None:
    events = read_records(EVENTS, Event)
    assert len({e.id for e in events}) == len(events) == 11
    assert all(frames.FIRST - timedelta(weeks=1) < e.day <= frames.LAST for e in events)
