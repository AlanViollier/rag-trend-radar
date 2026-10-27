"""rag-trend-radar: thin commands over the modules."""

from collections import Counter
from datetime import date
from functools import partial
from itertools import islice
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import typer
from rich import print as show
from rich.progress import track
from rich.table import Table

from rag_trend_radar import index
from rag_trend_radar.answer import answer
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed, token_counter
from rag_trend_radar.evals.questions import sample, write
from rag_trend_radar.evals.retrieval import rankings, score
from rag_trend_radar.ingest.assemble import assemble
from rag_trend_radar.ingest.chunk import pieces
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
def load(since: str = "2001-09-01", until: str = "2001-09-30") -> None:
    """Read the Enron emails sent from `since` to `until` into the store (default: the M1 slice)."""
    settings = get_settings()
    first, last = date.fromisoformat(since), date.fromisoformat(until)
    assembled = assemble(
        settings.data_dir / "enron" / "maildir",
        first,
        last,
        ZoneInfo(settings.timezone),
    )
    chunker = partial(
        pieces,
        count_tokens=token_counter(settings.embedding_model),
        boilerplate=assembled.boilerplate,
    )
    with db.connect() as conn:
        chunks = index.store(conn, assembled, chunker)
        index.record_days(conn, first, last, settings.timezone)
    kinds = Counter(d.kind for d in assembled.documents)
    show(
        f"{len(assembled.raw_files):,} files ({len(assembled.rejected):,} quarantined) → "
        f"{kinds['email']:,} emails + {kinds['quoted']:,} quoted messages → {chunks:,} chunks"
    )


@app.command("embed")
def embed_chunks(model: str = "", dims: int = 0, batch: int = 512) -> None:
    """Embed every chunk with no vector yet (default: the configured model and size)."""
    settings = get_settings()
    model, dims = model or settings.embedding_model, dims or settings.embedding_dims
    with db.connect() as conn:
        todo = list(index.unembedded(conn, model, dims).items())
        for start in track(range(0, len(todo), batch), description=f"{model} @ {dims}"):
            texts, ids = zip(*todo[start : start + batch], strict=True)
            vectors = embed(texts, model, dims)
            index.write_vectors(
                conn,
                model,
                [i for group in ids for i in group],
                np.repeat(vectors, [len(group) for group in ids], axis=0),
            )
            conn.commit()
    show(f"{sum(len(ids) for _, ids in todo):,} chunks embedded ({len(todo):,} distinct texts)")


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
def questions(emails: int = 150, quoted: int = 50, out: Path = QUESTIONS, seed: int = 1) -> None:
    """Have Claude write labelled questions from sampled emails and quoted messages (paid)."""
    settings = get_settings()
    with db.connect() as conn:
        day = last_day(conn)
        ## Half again as many sources as needed: Claude marks some unusable (automated notices).
        sources = {
            kind: sample(conn, kind, n * 3 // 2, seed)
            for kind, n in (("email", emails), ("quoted", quoted))
        }
    written = [
        q
        for kind, n in (("email", emails), ("quoted", quoted))
        for q in islice(write(sources[kind], day, settings.answer_model), n)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(q.model_dump_json() + "\n" for q in written))
    show(f"{len(written)} questions → {out}")


@app.command()
def recall(path: Path = QUESTIONS, depth: int = hybrid.DEPTH) -> None:
    """recall@k and MRR for every search method over the labelled questions (free, local)."""
    settings = get_settings()
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
        settings.embedding_model,
        settings.embedding_dims,
        query=True,
    )
    table = Table(
        title=f"{len(labelled)} questions, {settings.embedding_model} @ "
        f"{settings.embedding_dims}, depth {depth}"
    )
    table.add_column("method")
    for metric in METRICS:
        table.add_column(metric, justify="right")
    for method, results in rankings(labelled, vectors, depth).items():
        scores = score(relevant, results, METRICS)
        table.add_row(method, *(f"{scores[m]:.3f}" for m in METRICS))
    show(table)
