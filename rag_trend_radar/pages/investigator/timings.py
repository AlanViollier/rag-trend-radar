"""Page 2's search stage times, kept in the repo so the page is the same on every export.

Running a search again gives the same emails but never quite the same seconds. Each picked run keeps
the times its searches showed on the page it was approved with; a newly picked run keeps the ones
its first export measures.
"""

from pathlib import Path

from rag_trend_radar.records import Record, SearchStage

KEPT = Path("curated/investigator/search_times")


class SearchTimes(Record):
    """Each search of a run, in order: how long its stages took."""

    searches: tuple[tuple[SearchStage, ...], ...]


def kept(pick: str) -> SearchTimes | None:
    """Pick `pick`'s kept search times, or None when its page was never exported."""
    path = KEPT / f"{pick}.json"
    return SearchTimes.model_validate_json(path.read_text()) if path.exists() else None


def keep(pick: str, times: SearchTimes) -> None:
    """Keep `times` as pick `pick`'s search times."""
    KEPT.mkdir(parents=True, exist_ok=True)
    (KEPT / f"{pick}.json").write_text(times.model_dump_json(indent=1) + "\n")
