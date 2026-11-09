"""A run's own record: how long each step took, how much it handled, the memory it reached.

Every radar run saves one under data/reports/radar/, so any number quoted later traces back to a
run, its settings and its commit. A step that repeats (one per simulated day) keeps every sample,
so averages, medians and the slowest day can be read off afterwards.
"""

import resource
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_trend_radar.schema import RadarRun, RunStage, StageSummary


@dataclass
class Handled:
    """What a running step reports about itself: how many things it went through."""

    items: int = 0


class RunLog:
    """Collects one run's steps and results, then writes them to a file."""

    def __init__(self, kind: str, settings: dict[str, str | int | float]) -> None:
        self.kind = kind
        self.settings = settings
        self.started = datetime.now(UTC)
        self.results: dict[str, str | int | float] = {}
        self._stages: dict[str, list[RunStage]] = {}
        self._clock = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[Handled]:
        """Time one step; a step that fails is recorded too, with what it had got through."""
        handled = Handled()
        begun = time.perf_counter()
        try:
            yield handled
        finally:
            sample = RunStage(seconds=time.perf_counter() - begun, items=handled.items)
            self._stages.setdefault(name, []).append(sample)

    def run(self) -> RadarRun:
        """The run so far as a record."""
        return RadarRun(
            kind=self.kind,
            commit=commit(),
            started=self.started,
            seconds=time.perf_counter() - self._clock,
            peak_memory_mb=_peak_memory_mb(),
            settings=self.settings,
            stages=self._stages,
            results=self.results,
        )

    def save(self, folder: Path) -> Path:
        """Write the run to `folder` under its kind and start time; return the path."""
        path = folder / f"{self.kind}@{self.started:%Y%m%dT%H%M%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.run().model_dump_json(indent=1))
        return path


def summarise(samples: Sequence[RunStage]) -> StageSummary:
    """Totals and averages over a step's samples: one for a one-off, one a day for a replay."""
    seconds = np.array([s.seconds for s in samples])
    items = sum(s.items for s in samples)
    return StageSummary(
        runs=len(samples),
        total_seconds=float(seconds.sum()),
        mean_seconds=float(seconds.mean()),
        median_seconds=float(np.median(seconds)),
        p95_seconds=float(np.percentile(seconds, 95)),
        max_seconds=float(seconds.max()),
        items=items,
        ms_per_item=float(seconds.sum() * 1000 / items) if items else 0.0,
    )


def commit() -> str:
    """The code a run measured; "-dirty" when uncommitted changes took part."""
    done = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _peak_memory_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
