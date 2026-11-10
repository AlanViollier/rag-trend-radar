"""Every record that crosses a module boundary."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Frozen base: records are values, never mutated after construction."""

    model_config = ConfigDict(frozen=True)


class RawFile(Record):
    """One file of the dataset, exactly as read."""

    path: str
    mailbox: str
    folder: str
    content: bytes


class EmailHeaders(Record):
    """Who sent an email to whom: the `emails` row of a document."""

    message_id: str
    sender: str
    to_addrs: tuple[str, ...]
    cc_addrs: tuple[str, ...]
    bcc_addrs: tuple[str, ...]


class ParsedEmail(EmailHeaders):
    """The headers and body of a raw file, before cleaning."""

    sent_at: datetime
    subject: str
    body: str


class QuotedBlock(Record):
    """One quoted or forwarded message inside a body: its header lines, then its text."""

    header: str
    text: str


class SplitBody(Record):
    """An email body split into its own text and the messages it quotes, outermost first."""

    own: str
    quoted: tuple[QuotedBlock, ...]


class Document(Record):
    """A searchable thing ready to upsert, with who may see it and the files it came from.

    `author` and a known `sent_at` lead every chunk; a quoted message's date is when it was
    first quoted, not when it was written, so it leads with its author only.
    """

    source: str
    external_id: str
    kind: str
    sent_at: datetime
    title: str
    author: str
    body: str
    quoted: str
    public: bool
    readers: tuple[str, ...]
    paths: tuple[str, ...] = ()
    # A quoted message's readers are recounted from the emails quoting it (external ids).
    quoted_by: tuple[str, ...] = ()
    email: EmailHeaders | None = None
    date_uncertain: bool = False


class Copy(Record):
    """A file that parsed, as the scan saw it: the email it's a copy of, the date it claims."""

    path: str
    key: str
    sent_at: datetime
    own: str


class Rejected(Record):
    """A file set aside: one of a few reasons, and the value at fault (the date it gave)."""

    path: str
    reason: str
    detail: str


class Scan(Record):
    """Every file of the dataset read once: the copies that parsed and the ones set aside."""

    mailboxes: tuple[str, ...]
    copies: tuple[Copy, ...]
    rejected: tuple[Rejected, ...]


class DayFile(Record):
    """A file of one simulated day: as read, with its email's date resolved over all copies."""

    raw: RawFile
    key: str
    sent_at: datetime
    date_uncertain: bool


class SearchHit(Record):
    """One chunk returned by a search, best first."""

    chunk_id: int
    document_id: int
    text: str
    score: float
    context: str = ""  # what a short reply answered; `text` is only the reply's own words

    @property
    def full_text(self) -> str:
        """What the chunk was indexed and embedded from, and what a reranker should read."""
        return f"{self.text}\n\n{self.context}" if self.context else self.text


class Citation(Record):
    """A passage of a retrieved chunk that an answer relies on."""

    document_id: int
    cited_text: str


class Answer(Record):
    """A generated answer, its citations and what it cost."""

    question: str
    as_of: date
    text: str
    citations: tuple[Citation, ...]
    cost_usd: float


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


class EvalQuestion(Record):
    """A labelled question: asked by someone on a day, answered by known documents.

    Documents are named by external id (the content hash), which survives rebuilding the store;
    row ids don't.
    """

    id: str
    question: str
    as_of: date
    person: str
    relevant: tuple[str, ...]
    # The identifier a codes-style question quotes, as written; empty for the other styles.
    code: str = ""


class FactQuestion(Record):
    """A hard question: what its results must hold, must not hold, or that nothing should match."""

    id: str
    question: str
    as_of: date
    person: str = "investigator"
    facts: tuple[str, ...] = ()
    needs_mail: bool = False
    forbid: tuple[str, ...] = ()
    absent: bool = False


class FactResult(Record):
    """One hard question checked against one list of results."""

    id: str
    missing: tuple[str, ...]
    leaked: tuple[str, ...]
    top_score: float
    mail: int


class RadarEvent(Record):
    """Something that really happened, written down before the radar first ran: the page puts it
    on the timeline, and the naming pass notes which alerts caught it."""

    id: str
    day: date
    title: str
    words: tuple[str, ...]


class RunStage(Record):
    """One step of a radar run: how long it took and how many things it went through."""

    seconds: float
    items: int


class StageSummary(Record):
    """A step's samples added up: one sample for a one-off, one a day for a replay."""

    runs: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    max_seconds: float
    items: int
    ms_per_item: float


class RadarRun(Record):
    """A radar run saved whole, so every number quoted later traces back to it."""

    kind: str
    commit: str
    started: datetime
    seconds: float
    peak_memory_mb: float
    settings: dict[str, str | int | float]
    stages: dict[str, list[RunStage]]
    results: dict[str, str | int | float]


class RunManifest(Record):
    """What a recall run measured, so every number in the README traces back to a run."""

    questions: str
    count: int
    model: str
    dims: int
    depth: int
    ef_search: int
    reranker: str
    retype: str
    person: str = ""
    commit: str
    chunks: int
    started: datetime


class RecallRun(Record):
    """A recall run saved whole: re-scoring or judging it never re-runs a search.

    Rankings name documents by external id, best first, one entry per document (its best chunk);
    latency is per question, in milliseconds.
    """

    manifest: RunManifest
    rankings: dict[str, dict[str, list[str]]]
    latency_ms: dict[str, list[float]]
