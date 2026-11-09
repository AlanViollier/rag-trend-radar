from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.site import export, places
from rag_trend_radar.site.records import PageData, PageEvent

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
        recent_press=25,
        surge=10.0,
        moving=0.01,
    )


def _day(week: int, documents: int = 5) -> export.Day:
    return export.Day(ENDS[week], documents, documents * 2, 3, 4)


def _tables(fired: list[export.Fired] | None = None) -> export.Tables:
    seen = [_seen(1, 0), _seen(1, 2, mail=300), _seen(9, 1, shown=False), _seen(2, 3)]
    return export.Tables(seen, fired or [], [_day(0), _day(1, 7)], 100, 200)


@pytest.fixture(autouse=True)
def flat_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A map without UMAP, which needs more points than these tests have."""

    def spread(centres: NDArray[np.float32]) -> NDArray[np.float64]:
        return np.linspace(0.2, 0.8, 2 * len(centres)).reshape(-1, 2)

    monkeypatch.setattr(places, "map_points", spread)


def test_a_gap_is_filled_and_a_blob_runs_first_to_last_week() -> None:
    blob = export.page(_tables(), [], ENDS).blobs[0]
    assert (blob.id, blob.first, len(blob.mail), len(blob.x)) == (1, 0, 3, 3)


def test_press_is_a_share_of_recent_mail() -> None:
    assert export.page(_tables(), [], ENDS).blobs[0].press == (0.5, 0.5, 0.5)


def test_a_blob_only_an_alert_names_comes_last_with_a_name_and_no_lists() -> None:
    last = export.page(_tables(), [], ENDS).blobs[-1]
    assert (last.id, last.shown, last.mail) == (9, False, ())
    assert last.names == ((1, "gas pipeline storage"),)


def test_the_store_grows_from_what_came_before() -> None:
    weeks = export.page(_tables(), [], ENDS).weeks
    assert [w.documents for w in weeks[:3]] == [105, 112, 112]
    assert [w.chunks for w in weeks[:2]] == [210, 224]


def test_a_week_counts_only_its_own_mail() -> None:
    weeks = export.page(_tables(), [], ENDS).weeks
    assert [w.mail for w in weeks[:3]] == [5, 7, 0]


def test_live_blobs_counts_shown_blobs_only() -> None:
    assert [w.live_blobs for w in export.page(_tables(), [], ENDS).weeks] == [1, 1, 1, 1, 0, 0]


def test_an_alert_on_a_young_blob_is_new_on_an_older_one_speeding() -> None:
    young = export.Fired(ENDS[3], 1, 80.4, 2.0, ENDS[0])
    older = export.Fired(ENDS[4], 1, 90.0, 10.0, ENDS[0])
    alerts = export.page(_tables([young, older]), [], ENDS).alerts
    assert [(a.week, a.kind, a.observed) for a in alerts] == [(3, "new", 80), (4, "speeding", 90)]


def test_events_pass_through() -> None:
    event = PageEvent(id="x", day=ENDS[0], title="Something")
    assert export.page(_tables(), [event], ENDS).events == (event,)


def test_the_file_is_written_and_reads_back(tmp_path: Path) -> None:
    data = export.page(_tables(), [], ENDS)
    out = tmp_path / "docs" / "data.json"
    assert export.write(data, out) == out.stat().st_size
    assert PageData.model_validate_json(out.read_text()) == data


@pytest.mark.parametrize("number", ["123-45-6789", "4111 1111 1111 1111", "4111111111111111"])
def test_a_number_shaped_like_private_data_stops_the_export(tmp_path: Path, number: str) -> None:
    event = PageEvent(id="x", day=ENDS[0], title=f"Account {number}")
    out = tmp_path / "data.json"
    with pytest.raises(ExportRefused):
        export.write(export.page(_tables(), [event], ENDS), out)
    assert not out.exists()
