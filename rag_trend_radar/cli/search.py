"""The command a person asks through."""

from datetime import date

from rich import print as show

from rag_trend_radar.answer import answer
from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import last_day
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed
from rag_trend_radar.search import hybrid, pipeline


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
        ranked = pipeline.reranked(
            conn, question, query_vector, model, day, settings.rerank_model, hybrid.DEPTH
        )
        hits = pipeline.final(ranked, hybrid.DEPTH)
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
