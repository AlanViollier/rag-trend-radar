"""The agent's search: the pipeline `ask` runs, with optional dates and source."""

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, local_day, text
from rag_trend_radar.answer import search_result
from rag_trend_radar.config import get_settings
from rag_trend_radar.embed import embed
from rag_trend_radar.ingest.kinds import Source, source
from rag_trend_radar.search import hybrid, pipeline

DESCRIPTION = """Search the mail this person can read: keyword and meaning search, reranked, the \
best 10 chunks back as citable results, best first. Use it to find what someone wrote or said. \
Each result's title starts with its date and source (mail, press or bulk); its source names its \
document id, and read_email gives the whole email. Narrow with `since` / `until` when the \
question is about a time, and `source` "mail" for what people wrote themselves rather than \
forwarded press."""


class Input(ToolInput):
    """What Claude passes; field descriptions become the tool's schema."""

    query: str = Field(description="What to look for, in the words an email would use.")
    source: Literal["any", "mail", "press"] = Field(
        "any", description='"mail": people\'s own writing; "press": forwarded articles.'
    )
    since: date | None = Field(None, description="Only mail sent on or after this day.")
    until: date | None = Field(None, description="Only mail sent on or before this day.")


def run(desk: Desk, given: Input) -> ToolResult:
    """The best 10 chunks for `query` between the dates, as search_result blocks."""
    settings = get_settings()
    model = settings.embedding_model
    vector = embed([given.query], model, settings.embedding_dims, query=True)[0]
    keep = None if given.source == "any" else Source(given.source)
    ranked = pipeline.reranked(
        desk.conn,
        given.query,
        vector,
        model,
        desk.until(given.until),
        settings.rerank_model,
        hybrid.DEPTH,
        since=given.since,
        keep=keep,
    )
    hits = pipeline.final(ranked, hybrid.DEPTH)
    if not hits:
        return ToolResult([text("Nothing matches in the mail this person can read.")], "0 results")
    sent: dict[int, datetime] = dict(
        desk.conn.execute(
            "SELECT id, sent_at FROM documents WHERE id = ANY(%s)",
            ([hit.document_id for hit in hits],),
        ).fetchall()
    )
    blocks = []
    for hit in hits:
        # The API takes a tool result of search results only, no text beside them: the date and
        # source ride in each title, the document id in its `source`.
        block = search_result(hit)
        block["title"] = (
            f"{local_day(sent[hit.document_id])} · {source(hit.text)} · {block['title']}"
        )
        blocks.append(block)
    kinds = [source(hit.text) for hit in hits]
    summary = f"{len(hits)} results, {kinds.count(Source.MAIL)} of them people's own mail"
    return ToolResult(blocks, summary, tuple(dict.fromkeys(hit.document_id for hit in hits)))
