"""rag-trend-radar: thin commands over the modules."""

import sys
from datetime import date, timedelta
from itertools import islice
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
import typer
from psycopg.rows import TupleRow
from rich import print as show
from rich.progress import track
from rich.table import Table

from rag_trend_radar import batch, index
from rag_trend_radar.answer import answer
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed, token_counter
from rag_trend_radar.evals.questions import Style, sample, write
from rag_trend_radar.evals.retrieval import rankings, score
from rag_trend_radar.ingest.chunk import MAX_TOKENS, pieces
from rag_trend_radar.schema import EvalQuestion
from rag_trend_radar.search import hybrid
from rag_trend_radar.search.ranking import last_day
from rag_trend_radar.search.rerank import rerank

## How many reranked chunks Claude reads: enough for an answer spread over a thread, few enough
## that the bill stays at about a cent a question.
ANSWER_HITS = 10
QUESTIONS = Path("evals/questions.jsonl")
## recall@1: the answer is first. @10: Claude gets to read it. @50: the reranker gets to see it.
## MRR@10: 1 / the answer's rank, so second place counts half.
METRICS = ("recall@1", "recall@10", "recall@50", "mrr@10")

app = typer.Typer(no_args_is_help=True)


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
    """Run the daily loop for every day from `since` to `until` (default: the M1 slice).

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


@app.command()
def ask(question: str, person: str = "investigator", as_of: str = "") -> None:
    """Answer a question as `person` on day `as_of` (default: the last finished day).

    `person` is 'investigator' or an employee's mailbox (kean-s): row-level security decides
    what the search can see, before Claude sees anything.
    """
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    query_vector = embed([question], model, dims, query=True)[0]
    with db.connect(person) as conn:
        day = date.fromisoformat(as_of) if as_of else last_day(conn)
        candidates = hybrid.search(conn, question, query_vector, model, day, hybrid.DEPTH)
        hits = rerank(question, candidates, settings.rerank_model, top=ANSWER_HITS)
        if not hits:
            show(f"Nothing in {person}'s mail up to {day} matches.")
            return
        result = answer(question, hits, day, settings.answer_model)
        titles: dict[int, str] = dict(
            conn.execute(
                "SELECT id, title FROM documents WHERE id = ANY(%s)",
                ([c.document_id for c in result.citations],),
            ).fetchall()
        )
    show(f"[bold]{person}[/bold], as of {day}\n\n{result.text}\n")
    for n, document_id in enumerate(dict.fromkeys(c.document_id for c in result.citations), 1):
        show(f"[{n}] {titles[document_id]} (document {document_id})")
    show(f"\n${result.cost_usd:.4f}")


@app.command()
def questions(
    emails: int = 150,
    quoted: int = 50,
    out: Path = QUESTIONS,
    seed: int = 1,
    style: Style = Style.NAMES,
) -> None:
    """Have Claude write labelled questions from sampled emails and quoted messages (paid).

    `--style no-names` forbids naming people, companies, places or products, so keyword search
    can't lean on exact terms; `--style codes` quotes one deal or contract number as written,
    the case keyword search is there for. A different `--seed` samples different documents.
    """
    settings = get_settings()
    ## Spare sources: Claude marks some unusable (automated notices; for codes, mail whose only
    ## long numbers are phone numbers or zip codes).
    spare = 2 if style is Style.CODES else 3 / 2
    with db.connect() as conn:
        day = last_day(conn)
        sources = {
            kind: sample(conn, kind, int(n * spare), seed, style=style)
            for kind, n in (("email", emails), ("quoted", quoted))
        }
    written = [
        q
        for kind, n in (("email", emails), ("quoted", quoted))
        for q in islice(write(sources[kind], day, settings.answer_model, style=style), n)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(q.model_dump_json() + "\n" for q in written))
    show(f"{len(written)} questions → {out}")


@app.command()
def recall(
    path: Path = QUESTIONS,
    depth: int = hybrid.DEPTH,
    model: str = "",
    dims: int = 0,
    rerank_rows: bool = True,
    reranker: str = "",
) -> None:
    """recall@k and MRR for every search method over the labelled questions (free, local).

    `--model` and `--dims` score another embedding (embed it first), `--reranker` another
    cross-encoder; `--no-rerank-rows` skips the reranked rows, which take most of the time.
    """
    settings = get_settings()
    model, dims = model or settings.embedding_model, dims or settings.embedding_dims
    labelled = [EvalQuestion.model_validate_json(line) for line in path.read_text().splitlines()]
    with db.connect() as conn:
        ids: dict[str, int] = dict(
            conn.execute(
                "SELECT external_id, id FROM documents WHERE external_id = ANY(%s)",
                ([x for q in labelled for x in q.relevant],),
            ).fetchall()
        )
    relevant = {q.id: {ids[x] for x in q.relevant} for q in labelled}
    vectors = embed(
        [q.question for q in labelled],
        model,
        dims,
        query=True,
    )
    table = Table(title=f"{path.name}: {len(labelled)} questions, {model} @ {dims}, depth {depth}")
    table.add_column("method")
    for metric in METRICS:
        table.add_column(metric, justify="right")
    cross = (reranker or settings.rerank_model) if rerank_rows else None
    for method, results in rankings(labelled, vectors, model, depth, cross).items():
        scores = score(relevant, results, METRICS)
        table.add_row(method, *(f"{scores[m]:.3f}" for m in METRICS))
    show(table)


def _embed_missing(
    conn: psycopg.Connection[TupleRow],
    model: str,
    dims: int,
    batch_size: int = 512,
    *,
    progress: bool = False,
) -> int:
    ## A text embedded before, by any run in any schema, is copied from the cache; the rest
    ## embed once per distinct text. Commits per batch, so a stopped run resumes.
    index.from_cache(conn, model, dims)
    conn.commit()
    todo = list(index.unembedded(conn, model, dims).items())
    starts = range(0, len(todo), batch_size)
    for start in track(starts, description=f"{model} @ {dims}") if progress else starts:
        texts, ids = zip(*todo[start : start + batch_size], strict=True)
        vectors = embed(texts, model, dims)
        index.write_vectors(
            conn,
            model,
            [i for group in ids for i in group],
            np.repeat(vectors, [len(group) for group in ids], axis=0),
        )
        conn.commit()
    return sum(len(ids) for _, ids in todo)
