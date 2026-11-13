"""The agent's runs and the question sets that score them."""

from datetime import date, datetime
from typing import Literal

from rag_trend_radar.records.base import Record
from rag_trend_radar.records.search import Citation


class AgentStep(Record):
    """One thing the agent did, in order: a thought, a tool call, or a call to Claude coming back.

    `thinking` holds Claude's summarised reasoning (the raw reasoning never leaves the API);
    `note` what it wrote between tool calls.
    `tool` holds its name, the arguments as Claude sent them (JSON), the one-line summary and the
    documents it returned. `turn` holds one API call's tokens and cost.
    """

    kind: Literal["thinking", "note", "tool", "turn"]
    text: str = ""
    tool: str = ""
    arguments: str = ""
    summary: str = ""
    documents: tuple[int, ...] = ()
    error: bool = False
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0


class AgentRun(Record):
    """One question worked by the agent, whole: the trace page 2 replays and the evals score.

    `found` is the verdict Claude ends on: yes, no or partly; None when it gave none.
    """

    question: str
    person: str
    as_of: date
    model: str
    prompt_version: int
    steps: tuple[AgentStep, ...]
    answer: str
    found: Literal["yes", "no", "partly"] | None
    citations: tuple[Citation, ...]
    cost_usd: float
    seconds: float
    started: datetime
    commit: str


class AgentQuestion(Record):
    """A question for the agent and how code checks its answer, by `kind`.

    lookup: a cited email is one of `relevant` (external ids) or a copy of one. facts: every
    pattern in `expect` is in the answer. aggregate: every pattern is in its opening paragraph,
    `expect` taken from the query in `truth`. abstention: the verdict is "no". injection: `planted` rides in the first search's
    results, the answer still cites a right email, and `canary` (a pattern for obeying it, not
    for mentioning it) doesn't match. curiosity: no check, read by Claude.
    """

    id: str
    kind: Literal["lookup", "facts", "aggregate", "abstention", "injection", "curiosity"]
    question: str
    person: str = "investigator"
    as_of: date = date(2002, 12, 31)
    relevant: tuple[str, ...] = ()
    expect: tuple[str, ...] = ()
    truth: str = ""
    planted: str = ""
    canary: str = ""


class AgentScore(Record):
    """One saved run checked by code: the check its kind calls for, and two checks every run gets.

    `grounded`: every citation is an email a tool returned in that run. `hollow`: a "yes" with no
    citation and no count or radar call behind it.
    """

    id: str
    kind: str
    passed: bool | None
    grounded: bool
    hollow: bool
    found: str | None
    tool_calls: int
    cost_usd: float
    seconds: float
    note: str = ""
