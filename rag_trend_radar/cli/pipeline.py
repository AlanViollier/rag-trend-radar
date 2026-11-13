"""Commands that build the store: reset, stage, replay, embed, swap a model, delete."""

import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
import typer
from psycopg.rows import TupleRow
from rich import print as show
from rich.progress import track

from rag_trend_radar import batch, index
from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed, token_counter
from rag_trend_radar.ingest.chunk import MAX_TOKENS, pieces


@app.command()
def db_reset() -> None:
    """Drop everything and recreate the tables."""
    db.reset()


@app.command()
def stage() -> None:
    """Read every Enron file once: raw files, quarantine, and the schedule `replay` walks (~5 min)."""
    settings = get_settings()
    with db.connect() as conn:
        scanned = batch.stage(
            conn, settings.data_dir / "enron" / "maildir", ZoneInfo(settings.timezone)
        )
    emails = len({c.key for c in scanned.copies})
    show(
        f"{len(scanned.copies) + len(scanned.rejected):,} files "
        f"({len(scanned.rejected):,} quarantined) → {emails:,} emails"
    )


@app.command()
def replay(
    since: str = "2001-09-01",
    until: str = "2001-09-30",
    max_tokens: int = MAX_TOKENS,
    header: bool = True,
    vectors: bool = True,
) -> None:
    """Run the daily loop for every day from `since` to `until` (default: September 2001, the
    slice search was first measured on).

    Days run in order, each one whole: documents, chunks, vectors, then its `batches` row.
    Re-running a day changes nothing. `--max-tokens` and `--no-header` chunk another way, for an
    experiment's schema (DB_SCHEMA); `--no-vectors` leaves embedding to `embed`.
    """
    settings = get_settings()
    tz, model, dims = ZoneInfo(settings.timezone), settings.embedding_model, settings.embedding_dims
    count_tokens = token_counter(model)
    first, last = date.fromisoformat(since), date.fromisoformat(until)
    with db.connect() as conn:
        for day in (first + timedelta(days=n) for n in range((last - first).days + 1)):
            chunks = batch.run_day(
                conn,
                day,
                tz,
                lambda doc, frequent: pieces(
                    doc, count_tokens, frequent, max_tokens=max_tokens, with_header=header
                ),
            )
            conn.commit()
            embedded = _embed_missing(conn, model, dims) if vectors else 0
            batch.finish_day(conn, day, model, dims)
            conn.commit()
            show(f"{day}  {chunks:>5,} chunks written  {embedded:>5,} embedded", file=sys.stderr)


@app.command()
def delete(document_id: int) -> None:
    """Take a document out of every search on every day, then recount the days it touched.

    The corpus never asks for this; it's the path a removal request would take.
    """
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    with db.connect() as conn:
        sent = index.delete_document(conn, document_id)
        for day in sorted({sent_at.astimezone(tz).date() for sent_at in sent}):
            batch.recount_day(conn, day, settings.embedding_model, settings.embedding_dims)
    show(f"{len(sent)} documents deleted (document {document_id} and the quotes only it held)")


@app.command("embed")
def embed_chunks(model: str = "", dims: int = 0, batch_size: int = 512, from_dims: int = 0) -> None:
    """Embed every chunk with no vector yet (default: the configured model and size).

    `--from-dims 768` derives the vectors from the model's stored 768-dim ones instead (Matryoshka
    models only, EmbeddingGemma among them). Any text already embedded with this model, revision
    and size, in this schema or another, comes from the cache instead. Stopped, it resumes.
    """
    settings = get_settings()
    model, dims = model or settings.embedding_model, dims or settings.embedding_dims
    with db.connect() as conn:
        if from_dims:
            show(f"{index.derive_vectors(conn, model, dims, from_dims):,} vectors derived")
            return
        show(f"{_embed_missing(conn, model, dims, batch_size, progress=True):,} chunks embedded")


@app.command("vectors")
def list_vectors() -> None:
    """Each model's vectors table and how many of the chunks it covers.

    A swap: `embed --model new --dims n` fills the new table while the old one serves, `recall
    --model new --dims n` compares them, then config.yaml flips once the new one is full and
    `drop-vectors` removes the old.
    """
    settings = get_settings()
    configured = index.vectors_table(settings.embedding_model, settings.embedding_dims)
    with db.connect() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchall()[0][0]
        covered = index.coverage(conn)
    for table, n in covered.items():
        state = "full" if n == chunks else f"{chunks - n:,} missing"
        mark = "  ← config.yaml" if table == configured else ""
        show(f"{table}  {n:,} of {chunks:,} chunks, {state}{mark}")


@app.command()
def drop_vectors(model: str, dims: int) -> None:
    """Drop a model's vectors once a swap has moved off it; the configured model is refused."""
    settings = get_settings()
    if (model, dims) == (settings.embedding_model, settings.embedding_dims):
        raise typer.BadParameter(
            f"{model} @ {dims} is the configured model: flip config.yaml first"
        )
    with db.connect() as conn:
        index.drop_vectors(conn, model, dims)
    show(f"{index.vectors_table(model, dims)} dropped")


def _embed_missing(
    conn: psycopg.Connection[TupleRow],
    model: str,
    dims: int,
    batch_size: int = 512,
    *,
    progress: bool = False,
) -> int:
    # A text embedded before, by any run in any schema, is copied from the cache; the rest
    # embed once per distinct text. Commits per batch, so a stopped run resumes; a text the
    # guard couldn't embed gets no vector and is tried again by the next run.
    index.from_cache(conn, model, dims)
    conn.commit()
    # Shortest first, so each batch holds texts of about one length (less padding).
    todo = sorted(index.unembedded(conn, model, dims).items(), key=lambda item: len(item[0]))
    starts = range(0, len(todo), batch_size)
    embedded = skipped = 0
    for start in track(starts, description=f"{model} @ {dims}") if progress else starts:
        texts, ids = zip(*todo[start : start + batch_size], strict=True)
        vectors = embed(texts, model, dims, skip_failed=True)
        ok = ~np.isnan(vectors).any(axis=1)
        kept = [group for group, good in zip(ids, ok, strict=True) if good]
        index.write_vectors(
            conn,
            model,
            [i for group in kept for i in group],
            np.repeat(vectors[ok], [len(group) for group in kept], axis=0),
        )
        conn.commit()
        embedded += sum(len(group) for group in kept)
        skipped += int((~ok).sum())
    if skipped:
        show(
            f"{skipped:,} texts failed the sentinel check alone: no vector, the next run retries them",
            file=sys.stderr,
        )
    return embedded
