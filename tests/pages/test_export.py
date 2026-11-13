from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.pages.radar import export, places
from rag_trend_radar.pages.radar.records import AlertNote, Moment, Naming, PageData, PageEvent

START = date(2001, 1, 7)
ENDS = [START + timedelta(weeks=i) for i in range(6)]
WORDS = ("gas", "pipeline", "storage")


def _seen(blob: int, week: int, *, shown: bool = True, mail: int = 100) -> export.Seen:
    return export.Seen(
        blob=blob,
        shown=shown,
        week=ENDS[week],
        centre=np.ones(4, dtype=np.float32),
        words=WORDS,
        mail=mail,
        recent=50,
        surge=10.0,
    )


def _day(week: int, documents: int = 5) -> export.Day:
    return export.Day(ENDS[week], documents, documents * 2, 3)


def _tables(fired: list[export.Fired] | None = None) -> export.Tables:
    seen = [_seen(1, 0), _seen(1, 2, mail=300), _seen(9, 1, shown=False), _seen(2, 3)]
    dust = [export.Speck(np.ones(4, dtype=np.float32), ENDS[1])]
    return export.Tables(seen, fired or [], [_day(0), _day(1, 7)], 100, 200, dust)


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


def test_a_gap_is_filled_and_a_blob_runs_first_to_last_week() -> None:
    blob = export.page(_tables(), [], [], ENDS).blobs[0]
    assert (blob.id, blob.first, len(blob.mail), len(blob.x)) == (1, 0, 3, 3)


def test_a_blob_has_a_flat_and_a_lifted_place_every_week() -> None:
    blob = export.page(_tables(), [], [], ENDS).blobs[0]
    assert len(blob.x) == len(blob.lifted_z) == 3


def test_a_moment_fills_in_its_numbers_and_makes_its_alert_bold() -> None:
    fired = export.Fired(ENDS[3], 1, 1234.0, 2.4, ENDS[0])
    moment = Moment(day=ENDS[3], blob=1, text="{observed} against {normal}, {mail} that week")
    data = export.page(_tables([fired]), [], [moment], ENDS)
    assert (data.moments[0].week, data.moments[0].text) == (3, "1,234 against 2, 0 that week")
    assert (data.alerts[0].note, data.alerts[0].weight) == (data.moments[0].text, "bold")


def test_a_moment_says_how_long_after_its_event() -> None:
    fired = export.Fired(ENDS[3], 1, 10.0, 2.0, ENDS[0])
    event = PageEvent(id="x", day=ENDS[3] - timedelta(days=12), title="Something")
    moment = Moment(day=ENDS[3], blob=1, text="t", event="x")
    data = export.page(_tables([fired]), [event], [moment], ENDS)
    assert data.alerts[0].follows == "12 days after: Something"


def test_a_moment_naming_an_unknown_event_stops_the_export() -> None:
    fired = export.Fired(ENDS[3], 1, 10.0, 2.0, ENDS[0])
    with pytest.raises(ExportRefused):
        export.page(_tables([fired]), [], [Moment(day=ENDS[3], blob=1, text="t", event="x")], ENDS)


def _named(kind: Literal["topic", "newsletter"] = "topic", first: date = ENDS[0]) -> Naming:
    names = ((first, "Gas"), (ENDS[2], "Pipes"))
    return Naming(blob=1, first=first, kind=kind, family="Trading", names=names)


def test_a_read_blob_takes_its_names_and_family() -> None:
    blob = export.page(_tables(), [], [], ENDS, [_named()]).blobs[0]
    assert (blob.names, blob.family) == (((0, "Gas"), (2, "Pipes")), "Trading")


def test_a_blob_nobody_read_keeps_its_rough_name() -> None:
    blobs = {b.id: b for b in export.page(_tables(), [], [], ENDS, [_named()]).blobs}
    assert (blobs[2].family, len(blobs[2].names)) == ("", 1)


def test_a_blob_read_as_a_newsletter_leaves_the_map() -> None:
    blobs = {b.id: b for b in export.page(_tables(), [], [], ENDS, [_named("newsletter")]).blobs}
    assert (blobs[1].shown, blobs[2].shown) == (False, True)


def test_a_name_for_the_wrong_first_week_stops_the_export() -> None:
    with pytest.raises(ExportRefused):
        export.page(_tables(), [], [], ENDS, [_named(first=ENDS[1])])


def test_a_read_alert_takes_its_weight_and_note() -> None:
    fired = export.Fired(ENDS[3], 1, 80.0, 2.0, ENDS[0])
    note = AlertNote(day=ENDS[3], blob=1, weight="muted", headline="Gift ads", note="Junk.")
    alert = export.page(_tables([fired]), [], [], ENDS, read_alerts=[note]).alerts[0]
    assert (alert.weight, alert.headline, alert.note) == ("muted", "Gift ads", "Junk.")


def test_a_note_on_no_alert_stops_the_export() -> None:
    note = AlertNote(day=ENDS[3], blob=1, weight="bold", note="x")
    with pytest.raises(ExportRefused):
        export.page(_tables(), [], [], ENDS, read_alerts=[note])


def test_dust_arrives_the_week_it_was_sent() -> None:
    dust = export.page(_tables(), [], [], ENDS).dust
    assert (dust.first, len(dust.x), len(dust.lifted_z)) == ((1,), 1, 1)


def test_a_moment_with_no_alert_stops_the_export() -> None:
    with pytest.raises(ExportRefused):
        export.page(_tables(), [], [Moment(day=ENDS[3], blob=1, text="x")], ENDS)


def test_a_blob_not_on_the_main_map_comes_last_and_still_has_a_place() -> None:
    last = export.page(_tables(), [], [], ENDS).blobs[-1]
    assert (last.id, last.shown, len(last.mail), len(last.lifted_z)) == (9, False, 1, 1)


def test_the_store_grows_from_what_came_before() -> None:
    weeks = export.page(_tables(), [], [], ENDS).weeks
    assert [w.documents for w in weeks[:3]] == [105, 112, 112]
    assert [w.chunks for w in weeks[:2]] == [210, 224]


def test_a_week_counts_only_its_own_mail() -> None:
    weeks = export.page(_tables(), [], [], ENDS).weeks
    assert [w.mail for w in weeks[:3]] == [5, 7, 0]


def test_live_blobs_counts_shown_blobs_only() -> None:
    assert [w.live_blobs for w in export.page(_tables(), [], [], ENDS).weeks] == [1, 1, 1, 1, 0, 0]


def test_an_alert_on_a_young_blob_is_new_on_an_older_one_speeding() -> None:
    young = export.Fired(ENDS[3], 1, 80.4, 2.0, ENDS[0])
    older = export.Fired(ENDS[4], 1, 90.0, 10.0, ENDS[0])
    alerts = export.page(_tables([young, older]), [], [], ENDS).alerts
    assert [(a.week, a.kind, a.observed) for a in alerts] == [(3, "new", 80), (4, "speeding", 90)]


def test_small_floats_round_to_short_numbers() -> None:
    assert export._rounded(np.array([0.123], dtype=np.float32), 3) == (0.123,)


def test_events_pass_through() -> None:
    event = PageEvent(id="x", day=ENDS[0], title="Something")
    assert export.page(_tables(), [event], [], ENDS).events == (event,)


def test_the_page_carries_its_data_and_it_reads_back(tmp_path: Path) -> None:
    data = export.page(_tables(), [], [], ENDS)
    out = tmp_path / "docs" / "index.html"
    assert export.write(data, out) == out.stat().st_size
    held = out.read_text().split('type="application/json">')[1].split("</script>")[0]
    assert PageData.model_validate_json(held) == data


def test_data_cant_close_the_script_it_sits_in(tmp_path: Path) -> None:
    event = PageEvent(id="x", day=ENDS[0], title="a </script> b")
    out = tmp_path / "index.html"
    export.write(export.page(_tables(), [event], [], ENDS), out)
    assert out.read_text().count("</script>") == export.TEMPLATE.read_text().count("</script>")


@pytest.mark.parametrize("number", ["123-45-6789", "4111 1111 1111 1111", "4111111111111111"])
def test_a_number_shaped_like_private_data_stops_the_export(tmp_path: Path, number: str) -> None:
    event = PageEvent(id="x", day=ENDS[0], title=f"Account {number}")
    out = tmp_path / "data.json"
    with pytest.raises(ExportRefused):
        export.write(export.page(_tables(), [event], [], ENDS), out)
    assert not out.exists()
