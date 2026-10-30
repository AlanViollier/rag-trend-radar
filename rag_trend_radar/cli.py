"""rag-trend-radar: thin commands over the modules."""

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
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
from rag_trend_radar.evals import judge as judging
from rag_trend_radar.evals.leaks import check as check_leaks
from rag_trend_radar.evals.questions import Retype, Style, code_holds, retyped, sample, write
from rag_trend_radar.evals.retrieval import rankings, relevant, report
from rag_trend_radar.evals.review import review as write_review
from rag_trend_radar.ingest.chunk import MAX_TOKENS, pieces
from rag_trend_radar.schema import EvalQuestion, Judgment, RecallRun, RunManifest
from rag_trend_radar.search import hybrid
from rag_trend_radar.search.ranking import last_day
from rag_trend_radar.search.rerank import rerank

## How many reranked chunks Claude reads: enough for an answer spread over a thread, few enough
## that the bill stays at about a cent a question.
ANSWER_HITS = 10
QUESTIONS = Path("evals/questions.jsonl")
## The judge's verdicts per question file, kept in git so a reader can check them.
JUDGMENTS = Path("evals/judgments")
## hit_rate@1: an answer is first. @10: Claude gets to read one. @50: the reranker gets to see
## one. MRR@10: 1 / the first answer's rank, so second place counts half. Not recall@k: with
## judged labels a question can have several answers, and recall would count the share found.
METRICS = ("hit_rate@1", "hit_rate@10", "hit_rate@50", "mrr@10")

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
    ## long numbers are phone numbers or zip codes, or codes too many documents hold).
    spare = 3 if style is Style.CODES else 3 / 2
    with db.connect() as conn:
        day = last_day(conn)
        sources = {
            kind: sample(conn, kind, int(n * spare), seed, style=style)
            for kind, n in (("email", emails), ("quoted", quoted))
        }

    def kept(question: EvalQuestion) -> bool:
        if style is not Style.CODES:
            return True
        with db.connect(question.person) as conn:
            return code_holds(conn, question)

    written = [
        q
        for kind, n in (("email", emails), ("quoted", quoted))
        for q in islice(
            filter(kept, write(sources[kind], day, settings.answer_model, style=style)), n
        )
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
    ef_search: int = db.EF_SEARCH,
    retype: Retype | None = None,
    limit: int = 0,
) -> None:
    """recall@k and MRR for every search method over the labelled questions (free, local).

    `--model` and `--dims` score another embedding (embed it first), `--reranker` another
    cross-encoder; `--no-rerank-rows` skips the reranked rows. `--retype` asks the codes
    questions with their identifier typed another way; `--limit` keeps the first n questions,
    to time a run before committing to it. Every run is saved under data/reports/recall/; with
    judgments in evals/judgments/, a second table counts every document the judge found
    answering.
    """
    settings = get_settings()
    model, dims = model or settings.embedding_model, dims or settings.embedding_dims
    labelled = [EvalQuestion.model_validate_json(line) for line in path.read_text().splitlines()]
    labelled = labelled[:limit] if limit else labelled
    if retype is not None:
        labelled = [retyped(q, retype) for q in labelled if q.code]
    vectors = embed([q.question for q in labelled], model, dims, query=True)
    started = datetime.now(UTC)
    cross = (reranker or settings.rerank_model) if rerank_rows else None
    by_id, latency = rankings(labelled, vectors, model, depth, cross, ef_search)
    with db.connect() as conn:
        names: dict[int, str] = dict(
            conn.execute(
                "SELECT id, external_id FROM documents WHERE id = ANY(%s)",
                (list({d for docs in by_id.values() for ids in docs.values() for d in ids}),),
            ).fetchall()
        )
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchall()[0][0]
    ranked = {
        method: {qid: [names[d] for d in ids] for qid, ids in docs.items()}
        for method, docs in by_id.items()
    }
    run = RecallRun(
        manifest=RunManifest(
            questions=path.name,
            count=len(labelled),
            model=model,
            dims=dims,
            depth=depth,
            ef_search=ef_search,
            reranker=cross or "",
            retype=retype or "",
            commit=_commit(),
            chunks=chunks,
            started=started,
        ),
        rankings=ranked,
        latency_ms=latency,
    )
    out = settings.data_dir / "reports" / "recall" / f"{path.stem}@{started:%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run.model_dump_json())

    _scores(run, labelled)
    show(f"saved → {out}", file=sys.stderr)


@app.command()
def scores(run: Path) -> None:
    """Print a saved recall run's tables again, with the judgments as they are now (free)."""
    saved = RecallRun.model_validate_json(run.read_text())
    path = QUESTIONS.parent / saved.manifest.questions
    labelled = [EvalQuestion.model_validate_json(x) for x in path.read_text().splitlines()]
    asked = set(next(iter(saved.rankings.values())))
    _scores(saved, [q for q in labelled if q.id in asked])


def _scores(run: RecallRun, labelled: list[EvalQuestion]) -> None:
    m = run.manifest
    judged_path = JUDGMENTS / f"{Path(m.questions).stem}.jsonl"
    judgments = (
        [Judgment.model_validate_json(x) for x in judged_path.read_text().splitlines()]
        if judged_path.exists()
        else []
    )
    strict, judged = relevant(labelled, judgments)
    title = f"{m.questions}, {m.model} @ {m.dims}, depth {m.depth}, ef_search {m.ef_search}"
    if m.retype:
        title += f", codes retyped ({m.retype})"
    show(f"\n[bold]{title}[/bold]: {len(strict)} of {len(labelled)} questions, strict labels")
    show(report(strict, run.rankings, METRICS))
    if judgments:
        show("\n[bold]judged[/bold]: every document the judge found answering counts")
        show(report(judged, {m: run.rankings[m] for m in judging.JUDGED}, METRICS))
    timing = Table(title="milliseconds per question")
    for column in ("method", "median", "p95"):
        timing.add_column(column, justify="left" if column == "method" else "right")
    for method, times in run.latency_ms.items():
        timing.add_row(method, *(f"{np.percentile(times, p):.0f}" for p in (50, 95)))
    show(timing)


@app.command()
def review(run: Path) -> None:
    """Write the review page for a saved run and its judgments (free)."""
    saved = RecallRun.model_validate_json(run.read_text())
    stem = Path(saved.manifest.questions).stem
    labelled = [
        EvalQuestion.model_validate_json(x)
        for x in (QUESTIONS.parent / saved.manifest.questions).read_text().splitlines()
    ]
    judgments = [
        Judgment.model_validate_json(x)
        for x in (JUDGMENTS / f"{stem}.jsonl").read_text().splitlines()
    ]
    wanted = {j.external_id for j in judgments} | {
        docs[0] for docs in saved.rankings["hybrid + rerank"].values() if docs
    }
    with db.connect() as conn:
        texts = judging.documents(conn, wanted)
    out = get_settings().data_dir / "reports" / "review" / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(write_review(labelled, saved, judgments, texts))
    show(f"→ {out}")


@app.command()
def leaks(paths: list[Path]) -> None:
    """Every result checked for permission and time leaks over the labelled questions (free).

    Each question is asked by a mailbox that can't read its answer, then by its person on the
    day before its answer was sent; every document returned is checked as the owner.
    """
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    questions = [
        EvalQuestion.model_validate_json(line) for p in paths for line in p.read_text().splitlines()
    ]
    vectors = embed([q.question for q in questions], model, dims, query=True)
    found = check_leaks(questions, vectors, model, hybrid.DEPTH)
    table = Table(title=f"{len(questions)} questions, every result checked")
    for column in ("check", "results", "leaks"):
        table.add_column(column, justify="left" if column == "check" else "right")
    for name, n in found.checked.items():
        kind, method = name.split(", ")
        bad = sum(1 for x in found.leaks if (x.kind, x.method) == (kind, method))
        table.add_row(name, f"{n:,}", str(bad))
    show(table)
    for leak in found.leaks[:20]:
        show(leak)


@app.command()
def judge(path: Path, run: Path | None = None, dry_run: bool = False) -> None:
    """Have Claude judge each question's own document, and with `--run`, every document a method
    ranked above it in that saved run (paid, Batch API, cached).

    Writes evals/judgments/<questions>.jsonl, which `recall` reads for its judged table.
    `--dry-run` prints what would be sent and its cost, and sends nothing.
    """
    questions = [EvalQuestion.model_validate_json(x) for x in path.read_text().splitlines()]
    saved = RecallRun.model_validate_json(run.read_text()) if run else None
    with db.connect() as conn:
        todo = judging.pairs(conn, questions, saved)
    cost = judging.estimate(todo)
    show(
        f"{cost.pairs:,} pairs, {cost.cached:,} cached; to send: ~{cost.input_tokens:,} input"
        f" tokens, ~${cost.dollars:.2f}"
    )
    if dry_run:
        return
    judged = judging.judge(todo, lambda line: show(line, file=sys.stderr))
    if len(judged) < len(todo):
        show(f"{len(todo) - len(judged)} pairs got no verdict: run again to retry them")
    out = JUDGMENTS / f"{path.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(j.model_dump_json() + "\n" for j in judged))
    labels = {(q.id, x) for q in questions for x in q.relevant}
    own = [j for j in judged if (j.question_id, j.external_id) in labels]
    others = [j for j in judged if (j.question_id, j.external_id) not in labels]
    show(
        f"own document answers: {sum(j.answers for j in own)} of {len(own)};"
        f" other documents answering too: {sum(j.answers for j in others)} of {len(others)};"
        f" quotes not found: {sum(not j.quoted for j in judged if j.verdict != 'no')}"
    )
    show(f"→ {out}")


def _commit() -> str:
    ## The code a run measured; "-dirty" when uncommitted changes took part.
    done = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _embed_missing(
    conn: psycopg.Connection[TupleRow],
    model: str,
    dims: int,
    batch_size: int = 512,
    *,
    progress: bool = False,
) -> int:
    ## A text embedded before, by any run in any schema, is copied from the cache; the rest
    ## embed once per distinct text. Commits per batch, so a stopped run resumes; a text the
    ## guard couldn't embed gets no vector and is tried again by the next run.
    index.from_cache(conn, model, dims)
    conn.commit()
    ## Shortest first, so each batch holds texts of about one length (less padding).
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
