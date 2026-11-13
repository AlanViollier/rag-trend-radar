"""The agent's search: the pipeline `ask` runs, with optional dates and origin."""

import time
from datetime import date, datetime
from typing import Literal

from pydantic import Field

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult, text
from rag_trend_radar.answer import search_result
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import local_day
from rag_trend_radar.embed import embed
from rag_trend_radar.ingest.origin import Origin, origin
from rag_trend_radar.records import SearchStage
from rag_trend_radar.search import pipeline

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
    begun = time.perf_counter()
    vector = embed([given.query], settings.embedding_model, settings.embedding_dims, query=True)[0]
    embedded = SearchStage(name="embed", seconds=time.perf_counter() - begun, items=1)
    # The argument keeps the name Claude was given, `source`: the recorded runs call it so.
    wanted = None if given.source == "any" else Origin(given.source)
    ranked = pipeline.reranked(
        desk.conn, given.query, vector, desk.until(given.until), since=given.since, origin=wanted
    )
    stages = (embedded, *ranked.stages)
    fresh = [
        h for h in ranked.hits if h.chunk_id not in desk.shown and h.document_id not in desk.read
    ]
    hits = pipeline.final(fresh)
    again = [h.document_id for h in pipeline.final(ranked.hits) if h not in fresh]
    aside = (
        f"Left out, already shown in this run: documents {', '.join(map(str, dict.fromkeys(again)))}."
        if again
        else ""
    )
    if not hits:
        nothing = "Nothing new matches in the mail this person can read."
        return ToolResult([text(f"{nothing} {aside}".strip())], "0 new results", stages=stages)
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
        # origin ride in each title, the document id in its `source`.
        block = search_result(hit)
        block["title"] = (
            f"{local_day(sent[hit.document_id])} · {origin(hit.text)} · {block['title']}"
        )
        blocks.append(block)
    origins = [origin(hit.text) for hit in hits]
    summary = f"{len(hits)} results, {origins.count(Origin.MAIL)} of them people's own mail"
    if again:
        summary += f", {len(again)} repeats left out"
    found = tuple(dict.fromkeys(h.document_id for h in hits))
    return ToolResult(blocks, summary, found, aside, stages)
