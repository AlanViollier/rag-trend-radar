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
    body_hash: str
    public: bool
    readers: tuple[str, ...]
    paths: tuple[str, ...] = ()
    email: EmailHeaders | None = None
    date_uncertain: bool = False


class Assembled(Record):
    """A span of days read from the dataset, ready for the store."""

    mailboxes: tuple[str, ...]
    raw_files: tuple[RawFile, ...]
    rejected: dict[str, str]
    documents: tuple[Document, ...]
    boilerplate: frozenset[str]


class SearchHit(Record):
    """One chunk returned by a search, best first."""

    chunk_id: int
    document_id: int
    text: str
    score: float


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
    """A labelled question: asked by someone on a day, answered by known documents."""

    id: str
    question: str
    as_of: date
    person: str
    relevant_document_ids: tuple[int, ...]
