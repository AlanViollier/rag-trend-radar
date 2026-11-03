"""Commands that measure search: write questions, score them, check facts and leaks."""

import subprocess
import sys
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import numpy as np
from rich import print as show
from rich.table import Table

from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import last_day
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed
from rag_trend_radar.evals.copies import copies
from rag_trend_radar.evals.facts import check as check_facts
from rag_trend_radar.evals.leaks import check as check_leaks
from rag_trend_radar.evals.questions import Retype, Style, code_holds, retyped, sample, write
from rag_trend_radar.evals.retrieval import rankings, report
from rag_trend_radar.schema import EvalQuestion, FactQuestion, RecallRun, RunManifest
from rag_trend_radar.search import hybrid
from rag_trend_radar.search.collapse import collapse
from rag_trend_radar.search.mix import RESULTS, mix
from rag_trend_radar.search.rerank import rerank

QUESTIONS = Path("evals/questions.jsonl")
HARD_QUESTIONS = Path("evals/questions_hard.jsonl")
## hit_rate@1: an answer is first. @10: Claude gets to read one. @50: the reranker gets to see
## one. MRR@10: 1 / the first answer's rank, so second place counts half. Not recall@k: with
## judged labels a question can have several answers, and recall would count the share found.
METRICS = ("hit_rate@1", "hit_rate@10", "hit_rate@50", "mrr@10")


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
    person: str = "",
) -> None:
    """recall@k and MRR for every search method over the labelled questions (free, local).

    `--model` and `--dims` score another embedding (embed it first), `--reranker` another
    cross-encoder; `--no-rerank-rows` skips the reranked rows. `--retype` asks the codes
    questions with their identifier typed another way; `--limit` keeps the first n questions,
    to time a run before committing to it; `--person` asks every question as one person
    (investigator: the whole store) instead of a mailbox that can read the answer. Every run is saved under data/reports/recall/; a
    second table also counts copies of a label (forwards, re-sends) as right answers.
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
    by_id, latency = rankings(labelled, vectors, model, depth, cross, ef_search, person or None)
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
            person=person,
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
    strict = {q.id: set(q.relevant) for q in labelled}
    with db.connect() as conn:
        found = copies(conn, labelled, run.rankings.values())
    title = f"{m.questions}, {m.model} @ {m.dims}, depth {m.depth}, ef_search {m.ef_search}"
    if m.retype:
        title += f", codes retyped ({m.retype})"
    if m.person:
        title += f", asked as {m.person}"
    show(f"\n[bold]{title}[/bold]: {len(labelled)} questions, strict labels")
    show(report(strict, run.rankings, METRICS))
    extra = sum(len(found[q]) - len(strict[q]) for q in strict)
    show(f"\n[bold]copies count[/bold]: plus {extra} copies of a label, found by comparing text")
    show(report(found, run.rankings, METRICS))
    timing = Table(title="milliseconds per question")
    for column in ("method", "median", "p95"):
        timing.add_column(column, justify="left" if column == "method" else "right")
    for method, times in run.latency_ms.items():
        timing.add_row(method, *(f"{np.percentile(times, p):.0f}" for p in (50, 95)))
    show(timing)


@app.command()
def facts(path: Path = HARD_QUESTIONS, depth: int = hybrid.DEPTH) -> None:
    """Whether the results hold each hard question's facts, before and after collapse and caps (free).

    No answer is written and no API is called: code looks for the facts in the 10 chunks an
    answer would be written from.
    """
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    questions = [FactQuestion.model_validate_json(x) for x in path.read_text().splitlines()]
    vectors = embed([q.question for q in questions], model, dims, query=True)
    table = Table(title=f"{len(questions)} hard questions, 10 chunks each")
    for column in (
        "question",
        "raw",
        "ask",
        "mail raw → ask",
        "top score",
        "what's missing in ask",
    ):
        table.add_column(column, justify="right" if column == "top score" else "left")
    passed = {"raw": 0, "ask": 0}
    for q, vector_ in zip(questions, vectors, strict=True):
        with db.connect(q.person) as conn:
            candidates = hybrid.search(conn, q.question, vector_, model, q.as_of, depth)
        reranked = rerank(q.question, candidates, settings.rerank_model, top=depth)
        raw = check_facts(q, reranked[:RESULTS])
        ask_ = check_facts(q, mix(collapse(reranked, depth), RESULTS))
        for name, result in (("raw", raw), ("ask", ask_)):
            passed[name] += not (q.absent or result.missing or result.leaked)
        table.add_row(
            q.id,
            "-" if q.absent else "ok" if not (raw.missing or raw.leaked) else "no",
            "-" if q.absent else "ok" if not (ask_.missing or ask_.leaked) else "no",
            f"{raw.mail} → {ask_.mail}",
            f"{raw.top_score:.2f}",
            ", ".join((*ask_.missing, *ask_.leaked)),
        )
    show(table)
    checkable = sum(not q.absent for q in questions)
    show(f"passed: raw {passed['raw']}, ask {passed['ask']} of {checkable}")


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


def _commit() -> str:
    ## The code a run measured; "-dirty" when uncommitted changes took part.
    done = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()
