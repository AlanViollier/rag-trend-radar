"""Commands that build and read the radar."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from rich import print as show
from rich.table import Table

from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.run_log import RunLog, summarise
from rag_trend_radar.schema import RadarRun, TopicSummary
from rag_trend_radar.trends import assign, day, topics
from rag_trend_radar.trends.moves import Centres


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
        labels = topics.drop_thin(
            topics.cluster(points, min_size, min_samples, selection), pool.document_ids
        )
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
def radar_calibrate(
    since: str = "2000-01-01",
    until: str = "2000-05-31",
    unseen_until: str = "2000-06-30",
    hide: float = 0.1,
    joins: str = "0.70,0.75,0.78,0.80,0.82,0.84,0.86,0.88,0.90",
    min_size: int = 30,
    seed: int = 1,
) -> None:
    """How close a chunk should be to join a topic, on early mail only, so no hindsight (free).

    Clusters `since`..`until` with some placed chunks hidden from the centres, then for each join
    setting: do hidden chunks return to their own topic, how much mail no topic claimed gets pulled
    in, how much of the unseen month after joins at all.
    """
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    log = RunLog(
        "calibrate",
        {"since": since, "until": until, "unseen_until": unseen_until, "hide": hide, "seed": seed},
    )
    first, last = date.fromisoformat(since), date.fromisoformat(until)
    with log.stage("read chunks") as handled, db.connect() as conn:
        pool = topics.pool(conn, model, dims, first, last)
        unseen = topics.pool(
            conn, model, dims, last + timedelta(days=1), date.fromisoformat(unseen_until)
        )
        handled.items = len(pool.chunk_ids) + len(unseen.chunk_ids)
    with log.stage("squash and cluster") as handled:
        labels = topics.drop_thin(
            topics.cluster(topics.squash(pool.vectors, 5, 15, seed), min_size, 5),
            pool.document_ids,
        )
        handled.items = len(labels)
    placed = np.flatnonzero(labels != topics.NOISE)
    rng = np.random.default_rng(seed)
    hidden = rng.choice(placed, int(len(placed) * hide), replace=False)
    shown = labels.copy()
    shown[hidden] = topics.NOISE
    numbers, centres = assign.centres_of(pool.vectors, shown)
    own = pool.vectors @ centres.T
    member_rows = np.flatnonzero(shown != topics.NOISE)
    member = own[member_rows, np.searchsorted(numbers, shown[member_rows])]
    hidden_rows, hidden_best = assign.nearest(pool.vectors[hidden], centres)
    stray_rows = np.flatnonzero(labels == topics.NOISE)
    _, stray_best = assign.nearest(pool.vectors[stray_rows], centres)
    _, unseen_best = assign.nearest(unseen.vectors, centres)
    right = numbers[hidden_rows] == labels[hidden]
    log.results |= {
        "topics": len(numbers),
        "chunks": len(labels),
        "unassigned_share": round(float((labels == topics.NOISE).mean()), 4),
        "hidden": len(hidden),
        "unseen_chunks": len(unseen.chunk_ids),
    }
    ranges = Table(title="how close to the nearest centre")
    for column in ("chunks", "p5", "p10", "p25", "p50", "p75", "p90"):
        ranges.add_column(column, justify="left" if column == "chunks" else "right")
    for name, values in (
        ("topic members, own centre", member),
        ("hidden members, nearest", hidden_best),
        ("no topic claimed them", stray_best),
        ("unseen next month", unseen_best),
    ):
        cuts = np.percentile(values, (5, 10, 25, 50, 75, 90))
        ranges.add_row(name, *(f"{c:.3f}" for c in cuts))
        for p, c in zip((5, 10, 25, 50, 75, 90), cuts, strict=True):
            log.results[f"{name} p{p}"] = round(float(c), 4)
    show(ranges)
    table = Table(title="what each join setting does")
    for column in (
        "join",
        "hidden join",
        "of those, right topic",
        "hidden right overall",
        "strays pulled in",
        "unseen joins",
    ):
        table.add_column(column, justify="right")
    for value in (float(x) for x in joins.split(",")):
        joined = hidden_best >= value
        row = {
            "hidden_join": float(joined.mean()),
            "right_when_joined": float(right[joined].mean()) if joined.any() else 0.0,
            "hidden_right": float((joined & right).mean()),
            "strays_in": float((stray_best >= value).mean()),
            "unseen_join": float((unseen_best >= value).mean()),
        }
        for name, x in row.items():
            log.results[f"join {value:.2f} {name}"] = round(x, 4)
        table.add_row(f"{value:.2f}", *(f"{x:.1%}" for x in row.values()))
    show(table)
    show(
        " · ".join(
            f"{k} {log.results[k]}"
            for k in ("topics", "chunks", "unassigned_share", "hidden", "unseen_chunks")
        )
    )
    show(f"saved → {log.save(settings.data_dir / 'reports' / 'radar')}", file=sys.stderr)


@app.command()
def radar_replay(
    centres: Centres = Centres.FROZEN,
    join: float = 0.84,
    twin: float = 0.97,
    parent: float = 0.95,
    first: str = "2000-01-01",
    since: str = "2000-07-01",
    until: str = "2002-02-28",
    min_size: int = topics.MIN_SIZE,
    seed: int = 1,
) -> None:
    """Rebuild the radar's tables day by day and save the run (free, leaves the store alone).

    Topics start from `first` up to `since`; `--centres` is how they then behave. `--join` is how
    close a chunk must be to join a topic, `--twin` how close a new cluster must be to count as that
    topic, `--parent` how close to be born as its child. Prints each step's averages and each week.
    """
    settings = get_settings()
    log = RunLog(
        f"replay-{centres}",
        {
            "centres": centres.value,
            "join": join,
            "twin": twin,
            "parent": parent,
            "first": first,
            "since": since,
            "until": until,
            "min_size": min_size,
            "seed": seed,
            "model": settings.embedding_model,
        },
    )
    setup = day.Setup(
        centres=centres,
        join=join,
        twin=twin,
        parent=parent,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
        min_size=min_size,
        seed=seed,
    )
    db.reset_radar()
    with db.connect() as conn:
        day.replay(
            conn,
            setup,
            date.fromisoformat(first),
            date.fromisoformat(until),
            date.fromisoformat(since),
            log,
        )
    weeks = log.weeks
    chunks = sum(w.chunks for w in weeks)
    log.results |= {
        "weeks": len(weeks),
        "chunks": chunks,
        "joined_share": round(sum(w.joined for w in weeks) / chunks, 4) if chunks else 0.0,
        "topics_at_end": weeks[-1].topics if weeks else 0,
        "births": sum(w.births for w in weeks),
        "children": sum(w.children for w in weeks),
        "twins_absorbed": sum(w.twins_absorbed for w in weeks),
        "loose_clusters": sum(w.loose_clusters for w in weeks),
        "largest_share_max": max((w.largest_share for w in weeks), default=0.0),
        "near_twin_pairs_end": weeks[-1].near_twin_pairs if weeks else 0,
        "median_speed": round(float(np.median([w.median_speed for w in weeks])), 5) if weeks else 0,
        "moved_centres_share": round(
            sum(w.moved_centres for w in weeks) / max(sum(w.topics for w in weeks), 1), 4
        ),
    }
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
