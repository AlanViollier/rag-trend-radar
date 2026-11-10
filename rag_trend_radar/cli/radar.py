"""Commands that build and read the radar."""

import sys
from datetime import date, timedelta
from pathlib import Path

from rich import print as show
from rich.table import Table

from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.run_log import RunLog, summarise
from rag_trend_radar.schema import RadarRun
from rag_trend_radar.site import export
from rag_trend_radar.site.records import NAMES, AlertNote, Moment, Naming, PageEvent
from rag_trend_radar.trends import build, frames, topics

EVENTS = Path("evals/events.jsonl")
MOMENTS = Path("evals/moments.jsonl")
ALERT_NOTES = Path("evals/alert_notes.jsonl")
PAGE = Path("docs/radar/index.html")


@app.command()
def radar_reset() -> None:
    """Empty the radar's tables and leave the store alone: mail, chunks and vectors stay."""
    db.reset_radar()
    show("radar tables recreated, empty")


@app.command()
def radar_frames(first: str = "", last: str = "") -> None:
    """Make each week's map and blobs on both seeds, saved as files; resumes where it stopped (free).

    About 2 minutes a week per seed. Leaves the store alone: `radar-build` reads the files.
    """
    settings = get_settings()
    start = date.fromisoformat(first) if first else frames.FIRST
    end = date.fromisoformat(last) if last else frames.LAST
    log = RunLog(
        "frames", {"first": str(start), "last": str(end), "model": settings.embedding_model}
    )
    pool = _pool(log, end)
    ends = frames.week_ends(start, end)
    for seed in frames.SEEDS:
        frames.make(pool, ends, seed, _folder(seed), log)
    log.results["weeks"] = len(ends)
    path = log.save(settings.data_dir / "reports" / "radar")
    _show(log.run())
    show(f"saved → {path}", file=sys.stderr)


@app.command()
def radar_build() -> None:
    """Rebuild the blob tables and daily totals from the saved frames; changes nothing if rerun (free)."""
    settings = get_settings()
    log = RunLog("build", {"model": settings.embedding_model})
    ends = frames.week_ends()
    pool = _pool(log, frames.LAST)
    with log.stage("load frames") as handled:
        first, second = (frames.load(pool, ends, _folder(seed)) for seed in frames.SEEDS)
        handled.items = len(ends)
    with log.stage("build") as handled, db.connect() as conn:
        log.results |= build.write(conn, pool, first, second, settings.embedding_model)
        handled.items = len(ends)
    path = log.save(settings.data_dir / "reports" / "radar")
    _show(log.run())
    show(f"saved → {path}", file=sys.stderr)


@app.command()
def radar_export(out: Path = PAGE) -> None:
    """Write the replay page, data inside, from the radar tables (free, under a minute)."""
    settings = get_settings()
    log = RunLog("export", {"out": str(out)})
    with log.stage("read tables") as handled, db.connect(db.INVESTIGATOR) as conn:
        tables = export.read(
            conn, settings.embedding_model, settings.embedding_dims, frames.week_ends()
        )
        handled.items = len(tables.seen)
    with log.stage("smooth, place, name") as handled:
        data = export.page(
            tables,
            export.records_from(EVENTS, PageEvent),
            export.records_from(MOMENTS, Moment),
            frames.week_ends(),
            export.records_from(NAMES, Naming),
            export.records_from(ALERT_NOTES, AlertNote),
        )
        handled.items = len(data.blobs)
    with log.stage("write"):
        log.results["bytes"] = export.write(data, out)
    log.results |= {"weeks": len(data.weeks), "blobs": len(data.blobs), "alerts": len(data.alerts)}
    path = log.save(settings.data_dir / "reports" / "radar")
    _show(log.run())
    show(f"{out}, saved → {path}", file=sys.stderr)


@app.command()
def radar_runs(run: Path | None = None) -> None:
    """Every saved radar run, or one run's steps with their times and averages (free)."""
    if run is not None:
        _show(RadarRun.model_validate_json(run.read_text()))
        return
    table = Table(title="radar runs")
    for column in ("run", "commit", "seconds", "peak MB", "results"):
        table.add_column(column)
    for path in sorted((get_settings().data_dir / "reports" / "radar").glob("*.json")):
        saved = RadarRun.model_validate_json(path.read_text())
        results = " · ".join(f"{k} {v}" for k, v in saved.results.items())
        table.add_row(
            path.stem,
            saved.commit,
            f"{saved.seconds:,.0f}",
            f"{saved.peak_memory_mb:,.0f}",
            results,
        )
    show(table)


def _show(run: RadarRun) -> None:
    table = Table(title=f"{run.kind}, {run.seconds:,.1f} s, peak {run.peak_memory_mb:,.0f} MB")
    for column in (
        "step",
        "runs",
        "total s",
        "mean s",
        "median s",
        "p95 s",
        "max s",
        "items",
        "ms/item",
    ):
        table.add_column(column, justify="left" if column == "step" else "right")
    for name, samples in run.stages.items():
        s = summarise(samples)
        table.add_row(
            name,
            str(s.runs),
            f"{s.total_seconds:,.1f}",
            f"{s.mean_seconds:,.2f}",
            f"{s.median_seconds:,.2f}",
            f"{s.p95_seconds:,.2f}",
            f"{s.max_seconds:,.2f}",
            f"{s.items:,}",
            f"{s.ms_per_item:,.2f}",
        )
    show(table)
    show(" · ".join(f"{k} {v}" for k, v in {**run.settings, **run.results}.items()))


def _pool(log: RunLog, last: date) -> topics.Pool:
    settings = get_settings()
    since = frames.FIRST - timedelta(weeks=frames.WEEKS)
    with log.stage("read chunks") as handled, db.connect() as conn:
        pool = topics.pool(conn, settings.embedding_model, settings.embedding_dims, since, last)
        handled.items = len(pool.chunk_ids)
    return pool


def _folder(seed: int) -> Path:
    return get_settings().data_dir / "radar" / "frames" / f"seed{seed}"
