"""The real events the page puts on its timeline: they load and sit inside the radar's run."""

from datetime import date
from pathlib import Path

from rag_trend_radar.records import RadarEvent

RADAR_STARTS, RADAR_ENDS = date(2000, 7, 1), date(2002, 2, 28)


def test_every_event_sits_in_the_radars_window() -> None:
    lines = Path("curated/radar/events.jsonl").read_text().splitlines()
    events = [RadarEvent.model_validate_json(line) for line in lines]
    assert len({e.id for e in events}) == len(events) == 11
    assert all(RADAR_STARTS <= e.day <= RADAR_ENDS for e in events)
