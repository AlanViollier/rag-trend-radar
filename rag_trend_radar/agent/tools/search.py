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
document id, and read_email gives the whole email. A run never gets the same chunk twice: \
results already shown are left out and named, and newer ones take their place. Narrow with `since` / `until` when the \
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
    fresh = [h for h in ranked if h.chunk_id not in desk.shown and h.document_id not in desk.read]
    hits = pipeline.final(fresh, hybrid.DEPTH)
    again = [h.document_id for h in pipeline.final(ranked, hybrid.DEPTH) if h not in fresh]
    aside = (
        f"Left out, already shown in this run: documents {', '.join(map(str, dict.fromkeys(again)))}."
        if again
        else ""
    )
    if not hits:
        nothing = "Nothing new matches in the mail this person can read."
        return ToolResult([text(f"{nothing} {aside}".strip())], "0 new results")
    desk.shown.update(hit.chunk_id for hit in hits)
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
    if again:
        summary += f", {len(again)} repeats left out"
    return ToolResult(blocks, summary, tuple(dict.fromkeys(h.document_id for h in hits)), aside)
