"""Saved runs, so every number quoted later traces back to one."""

from datetime import datetime

from rag_trend_radar.records.base import Record


class RunStage(Record):
    """One step of a radar run: how long it took and how many things it went through."""

    seconds: float
    items: int


class StageSummary(Record):
    """A step's samples added up: one sample for a one-off, one a day for a replay."""

    runs: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    max_seconds: float
    items: int
    ms_per_item: float


class RadarRun(Record):
    """A radar run saved whole, so every number quoted later traces back to it."""

    kind: str
    commit: str
    started: datetime
    seconds: float
    peak_memory_mb: float
    settings: dict[str, str | int | float]
    stages: dict[str, list[RunStage]]
    results: dict[str, str | int | float]


class RunManifest(Record):
    """What a recall run measured, so every number in the README traces back to a run."""

    questions: str
    count: int
    model: str
    dims: int
    depth: int
    ef_search: int
    reranker: str
    retype: str
    person: str = ""
    commit: str
    chunks: int
    started: datetime


class RecallRun(Record):
    """A recall run saved whole: re-scoring it never re-runs a search.

    Rankings name documents by external id, best first, one entry per document (its best chunk);
    latency is per question, in milliseconds.
    """

    manifest: RunManifest
    rankings: dict[str, dict[str, list[str]]]
    latency_ms: dict[str, list[float]]
