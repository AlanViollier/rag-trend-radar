"""A saved run's steps for the page, its tool calls run again on the store: free, no call to Claude.

The trace keeps what Claude was told in one line per call; running each call again, in order, on the
same person and day gives what the page shows inside it (the hits, the email, the counts). A call
that now returns other emails than the run saw stops the export: the page would show the wrong mail.
"""

import json
from collections.abc import Sequence

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.agent import Desk
from rag_trend_radar.agent import tools as agent_tools
from rag_trend_radar.agent.desk import ToolResult
from rag_trend_radar.answer import SOURCE_PREFIX
from rag_trend_radar.evals.agent import PLANTED
from rag_trend_radar.exceptions import ExportRefused, ToolRefused
from rag_trend_radar.pages.investigator.records import Hit, Note, Step, Thought, ToolCall, Turn
from rag_trend_radar.records import AgentRun, AgentStep

# A search result's title is "<day> · <origin> · <subject>" (agent/tools/search.py).
TITLE_SEPARATOR = " · "
# Where the harness planted its email among the first search's results (agent/loop.py).
PLANTED_AT = 1
# Times kept to the millisecond: a float's 17 digits read as a card number to the export's check.
DIGITS = 3


def steps(conn: psycopg.Connection[TupleRow], run: AgentRun, staged: bool) -> list[Step]:
    """Every step of `run` as the page plays it, on a connection set to the run's person.

    `staged`: the run had an email planted in its first search, shown among that search's hits.
    """
    desk = Desk(conn, run.person, run.as_of)
    out: list[Step] = []
    planting = staged
    for step in run.steps:
        if step.kind == "thinking":
            out.append(Thought(text=step.text.strip()))
        elif step.kind == "note":
            out.append(Note(text=step.text.strip()))
        elif step.kind == "turn":
            out.append(Turn(seconds=round(step.seconds, DIGITS)))
        else:
            call = _call(desk, step, planting)
            planting = planting and not call.hits
            out.append(call)
    return out


def named(steps: Sequence[Step]) -> set[int]:
    """Every email the steps show: search hits and emails read."""
    found: set[int] = set()
    for step in steps:
        if isinstance(step, ToolCall):
            found |= {h.document_id for h in step.hits} | ({step.read} if step.read else set())
    return found


def _call(desk: Desk, step: AgentStep, planting: bool) -> ToolCall:
    arguments = json.loads(step.arguments or "{}")
    recorded = ToolCall(
        tool=step.tool,
        arguments=arguments,
        summary=step.summary,
        seconds=round(step.seconds, DIGITS),
    )
    try:
        done = agent_tools.call(desk, step.tool, arguments)
    except ToolRefused as error:
        return recorded.model_copy(update={"text": str(error)})
    if done.documents != step.documents:
        raise ExportRefused(
            f"{step.tool} {step.arguments} now returns other emails than the run saw: run it again"
        )
    if step.tool == "search":
        hits = _hits(done)
        if planting and hits:
            hits.insert(PLANTED_AT, Hit(document_id=PLANTED, origin="planted"))
        stages = tuple(
            s.model_copy(update={"seconds": round(s.seconds, DIGITS)}) for s in done.stages
        )
        return recorded.model_copy(update={"stages": stages, "hits": tuple(hits)})
    if step.tool == "read_email":
        return recorded.model_copy(update={"read": done.documents[0]})
    return recorded.model_copy(
        update={"text": "\n".join(b["text"] for b in done.blocks if b["type"] == "text")}
    )


def _hits(done: ToolResult) -> list[Hit]:
    return [
        Hit(
            document_id=int(block["source"].removeprefix(SOURCE_PREFIX)),
            origin=block["title"].split(TITLE_SEPARATOR)[1],
        )
        for block in done.blocks
        if block["type"] == "search_result"
    ]
