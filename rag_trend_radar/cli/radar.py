"""Commands that build and read the radar."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
from rich import print as show
from rich.table import Table

from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.run_log import RunLog, summarise
from rag_trend_radar.schema import RadarRun, TopicSummary
from rag_trend_radar.trends import topics


@app.command()
def radar_reset() -> None:
    """Empty the radar's tables and leave the store alone: mail, chunks and vectors stay."""
    db.reset_radar()
    show("radar tables recreated, empty")


@app.command()
def radar_cluster(
    since: str = "2000-01-01",
    until: str = "2000-06-30",
    sample: int = 0,
    dims: int = 0,
    min_size: int = 15,
    min_samples: int = 5,
    selection: str = "eom",
    seed: int = 1,
    squash: int = 0,
    neighbors: int = 15,
) -> None:
    """Cluster a period's chunks and save the topics to read, without touching the store (free).

    `--sample` clusters that many random chunks, to time a size before committing to it;
    `--dims` shortens the vectors first; `--squash` clusters a UMAP map of that many numbers,
    `--neighbors` being how many nearby chunks shape it. Prints and saves every step's time and the peak memory.
    """
    settings = get_settings()
    model, full = settings.embedding_model, settings.embedding_dims
    log = RunLog(
        "cluster",
        {
            "since": since,
            "until": until,
            "sample": sample,
            "dims": dims or full,
            "min_size": min_size,
            "min_samples": min_samples,
            "selection": selection,
            "seed": seed,
            "squash": squash,
            "neighbors": neighbors,
            "model": model,
        },
    )
    with log.stage("read chunks") as handled, db.connect() as conn:
        pool = topics.pool(conn, model, full, date.fromisoformat(since), date.fromisoformat(until))
        handled.items = len(pool.chunk_ids)
    log.results["chunks_in_period"] = len(pool.chunk_ids)
    if sample and sample < len(pool.chunk_ids):
        rows = np.random.default_rng(seed).choice(len(pool.chunk_ids), sample, replace=False)
        pool = pool.take(np.sort(rows))
    vectors = topics.shorten(pool.vectors, dims) if dims else pool.vectors
    points = vectors
    if squash:
        with log.stage("squash") as handled:
            points = topics.squash(vectors, squash, neighbors, seed)
            handled.items = len(points)
    with log.stage("cluster") as handled:
        labels = topics.cluster(points, min_size, min_samples, selection)
        handled.items = len(labels)
    with log.stage("describe") as handled:
        found = topics.describe(pool, vectors, labels)
        handled.items = len(found)
    noise = float((labels == topics.NOISE).mean())
    log.results |= {
        "chunks_clustered": len(labels),
        "topics": len(found),
        "noise_share": round(noise, 4),
        "largest_topic_share": round(max((f.chunks for f in found), default=0) / len(labels), 4),
        "median_topic_chunks": int(np.median([f.chunks for f in found])) if found else 0,
        "mean_cohesion": round(float(np.mean([f.cohesion for f in found])), 4) if found else 0.0,
    }
    log.topics = tuple(
        TopicSummary(
            label=f.label,
            chunks=f.chunks,
            documents=f.documents,
            cohesion=round(f.cohesion, 4),
            keywords=f.keywords,
            nearest=f.nearest,
        )
        for f in found
    )
    path = log.save(settings.data_dir / "reports" / "radar")
    _show(log.run())
    show(f"saved → {path}", file=sys.stderr)


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
