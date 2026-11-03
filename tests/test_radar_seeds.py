"""The frozen answer key and worries: they load, and the scoreboard can run on them."""

from datetime import date
from pathlib import Path

from rag_trend_radar.schema import RadarEvent, WatchlistSeed

RADAR_STARTS, RADAR_ENDS = date(2000, 7, 1), date(2002, 2, 28)


def test_every_event_sits_in_the_radars_window_and_can_be_matched() -> None:
    lines = Path("evals/events.jsonl").read_text().splitlines()
    events = [RadarEvent.model_validate_json(line) for line in lines]
    assert len({e.id for e in events}) == len(events) == 11
    assert all(RADAR_STARTS <= e.day <= RADAR_ENDS for e in events)
    assert all(len(set(e.words)) >= 2 for e in events)


def test_every_worry_has_a_meaning_and_words_to_match() -> None:
    lines = Path("evals/watchlists.jsonl").read_text().splitlines()
    seeds = [WatchlistSeed.model_validate_json(line) for line in lines]
    assert len({s.name for s in seeds}) == len(seeds) > 0
    assert all(s.query and s.terms for s in seeds)
