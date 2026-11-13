"""Mail on its way into the store: files, parsed emails, split bodies, documents."""

from datetime import datetime

from rag_trend_radar.records.base import Record


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
