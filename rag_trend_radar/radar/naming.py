"""The naming pass: every cluster read by a person, its names over time and what kind it is.

Read with hindsight, so the page and the agent's tools use it and the alerts never do. A name shows
only from the week it was given, so a topic isn't called what it later became.
"""

from collections.abc import Sequence
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

from rag_trend_radar.records import Record, read_records

NAMES = Path("curated/radar/cluster_names.jsonl")


class Naming(Record):
    """A cluster as a person read it. `names` are (week it starts, name); only a `topic` goes on
    the map."""

    cluster: int
    first: date
    kind: Literal["topic", "newsletter", "footer", "chatter"]
    family: str
    names: tuple[tuple[date, str], ...]


def readings() -> list[Naming]:
    """Every cluster the naming pass read, in file order."""
    return read_records(NAMES, Naming)


def name(cluster: int, week: date, keywords: Sequence[str]) -> str:
    """The name a cluster had by `week`, or its top keywords when it has none yet."""
    known = every_name(cluster, week)
    return known[-1][1] if known else ", ".join(keywords[:4])


def every_name(cluster: int, until: date) -> list[tuple[date, str]]:
    """Every name a cluster had up to `until`, with the week it started."""
    reading = _by_cluster().get(cluster)
    return [(start, label) for start, label in reading.names if start <= until] if reading else []


def kind(cluster: int, built: str) -> str:
    """Topic, newsletter, footer or chatter: the naming pass's reading, else the build's rule."""
    reading = _by_cluster().get(cluster)
    return reading.kind if reading else built


@cache
def _by_cluster() -> dict[int, Naming]:
    return {reading.cluster: reading for reading in readings()}
