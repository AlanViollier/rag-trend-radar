"""Read a span of days from the Enron files: raw files, quarantine, and the documents they make.

Every file is scanned, not only the span's: copies of one email can sit hours apart across the
span's edges, and boilerplate is counted over all mail sent up to its last day.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.ingest.clean import (
    BOILERPLATE_MIN_EMAILS,
    block_author,
    block_subject,
    frequent_lines,
    normalize,
    split_quoted,
)
from rag_trend_radar.ingest.dedup import content_hash, resolve_sent_at, text_hash
from rag_trend_radar.schema import (
    Assembled,
    Document,
    EmailHeaders,
    ParsedEmail,
    QuotedBlock,
    SplitBody,
)
from rag_trend_radar.sources import enron

SOURCE = "enron"
## Export damage shifts a copy's date by up to ~12 hours: copies within a day of the span are
## kept whole, since one of them may be the copy whose date wins.
_MARGIN = timedelta(days=1)


def assemble(root: Path, since: date, until: date, tz: ZoneInfo) -> Assembled:
    """The documents of every email sent from `since` to `until`, inclusive, in local days.

    Emails come sorted by (sent_at, content hash), so runs reproduce exactly. A quoted message
    whose text is no email's own text in the span becomes a document of its own.
    """
    start = datetime.combine(since, time(), tz)
    end = datetime.combine(until + timedelta(days=1), time(), tz)
    scan = partial(_scan, root=root, window=(start - _MARGIN, end + _MARGIN))
    rejected: dict[str, str] = {}
    copies: dict[str, list[_Copy]] = defaultdict(list)
    with ProcessPoolExecutor() as pool:
        paths = sorted(p for p in root.rglob("*") if p.is_file())
        for scanned in pool.map(scan, paths, chunksize=500):
            if isinstance(scanned, _Rejected):
                rejected[scanned.path] = scanned.reason
            else:
                copies[scanned.key].append(scanned)

    dates = {key: resolve_sent_at(c.sent_at for c in group) for key, group in copies.items()}
    boilerplate = frequent_lines(
        (group[0].own for key, group in copies.items() if dates[key][0] < end),
        BOILERPLATE_MIN_EMAILS,
    )
    in_span = sorted(
        (key for key in copies if start <= dates[key][0] < end), key=lambda k: (dates[k][0], k)
    )
    emails = [_email(key, copies[key], *dates[key]) for key in in_span]
    paths_kept = sorted({c.path for key in in_span for c in copies[key]} | rejected.keys())
    return Assembled(
        mailboxes=tuple(sorted(p.name for p in root.iterdir() if p.is_dir())),
        raw_files=tuple(enron.read_raw_file(root, root / path) for path in paths_kept),
        rejected=rejected,
        documents=(*(doc for doc, _ in emails), *_orphans(emails)),
        boilerplate=boilerplate,
    )


## Scanning, one file per worker call.


@dataclass(frozen=True)
class _Copy:
    path: str
    mailbox: str
    key: str
    sent_at: datetime
    own: str
    ## Only copies near the span keep their parse: the rest are needed for dates and boilerplate.
    parsed: tuple[ParsedEmail, SplitBody] | None


@dataclass(frozen=True)
class _Rejected:
    path: str
    reason: str


def _scan(path: Path, root: Path, window: tuple[datetime, datetime]) -> _Copy | _Rejected:
    raw = enron.read_raw_file(root, path)
    try:
        email = enron.parse(raw)
    except ParseError as error:
        return _Rejected(raw.path, str(error))
    split = split_quoted(normalize(email.body))
    near = window[0] <= email.sent_at < window[1]
    return _Copy(
        path=raw.path,
        mailbox=raw.mailbox,
        key=content_hash(email.subject, email.body),
        sent_at=email.sent_at,
        own=split.own,
        parsed=(email, split) if near else None,
    )


## Building documents.


def _email(
    key: str, group: list[_Copy], sent_at: datetime, uncertain: bool
) -> tuple[Document, SplitBody]:
    email, split = next(c.parsed for c in group if c.parsed is not None)
    quoted = "\n\n".join(f"{b.header}\n\n{b.text}".strip() for b in split.quoted)
    document = Document(
        source=SOURCE,
        external_id=key,
        kind="email",
        sent_at=sent_at,
        title=email.subject,
        author=email.sender,
        body=split.own,
        quoted=quoted,
        body_hash=text_hash(split.own + "\x00" + quoted),
        public=False,
        readers=tuple(sorted({c.mailbox for c in group})),
        paths=tuple(sorted(c.path for c in group)),
        email=EmailHeaders.model_validate(email.model_dump(include=set(EmailHeaders.model_fields))),
        date_uncertain=uncertain,
    )
    return document, split


def _orphans(emails: list[tuple[Document, SplitBody]]) -> list[Document]:
    ## Emails arrive in (sent_at, hash) order, so a quoted message is dated by its first quote.
    own = {text_hash(doc.body) for doc, _ in emails if doc.body}
    first: dict[str, tuple[Document, QuotedBlock]] = {}
    readers: dict[str, set[str]] = defaultdict(set)
    for doc, split in emails:
        for block in split.quoted:
            digest = text_hash(block.text)
            if not block.text or digest in own:
                continue
            first.setdefault(digest, (doc, block))
            readers[digest].update(doc.readers)
    return [
        Document(
            source=SOURCE,
            external_id=digest,
            kind="quoted",
            sent_at=doc.sent_at,
            title=block_subject(block.header) or doc.title,
            author=block_author(block.header),
            body=block.text,
            quoted="",
            body_hash=digest,
            public=False,
            readers=tuple(sorted(readers[digest])),
        )
        for digest, (doc, block) in first.items()
    ]
