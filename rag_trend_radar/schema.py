"""Every record that crosses a module boundary."""

from datetime import date, datetime

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
    ## A quoted message's readers are recounted from the emails quoting it (external ids).
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
    ## The identifier a codes-style question quotes, as written; empty for the other styles.
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
    """Something that really happened, written down before the radar first ran: the scoreboard's
    answer key. An alert claims it when two of `words` are among its topic's keywords."""

    id: str
    day: date
    title: str
    words: tuple[str, ...]


class WatchlistSeed(Record):
    """An investigator's saved worry: `query` is matched by meaning, `terms` as written."""

    name: str
    query: str
    terms: tuple[str, ...]


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


class TopicSummary(Record):
    """A topic as a run found it, to read: size, how tight it is, its words, the mail at its centre."""

    label: int
    chunks: int
    documents: int
    cohesion: float
    keywords: tuple[str, ...]
    nearest: tuple[str, ...]


class RadarWeek(Record):
    """One week of a radar replay, to compare centre versions by."""

    day: date
    topics: int
    births: int
    children: int
    twins_absorbed: int
    chunks: int
    joined: int
    moved_centres: int
    largest_share: float
    near_twin_pairs: int
    median_speed: float
    loose_clusters: int = 0


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
    topics: tuple[TopicSummary, ...] = ()
    weeks: tuple[RadarWeek, ...] = ()


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
