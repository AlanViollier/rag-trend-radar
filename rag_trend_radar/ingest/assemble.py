"""Read the Enron files once for the replay's schedule, and turn one day's files into documents.

Every file is scanned before any day runs: copies of one email can sit hours apart across a
day's edges, so an email's date is only known once all its copies are seen.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
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
from rag_trend_radar.records import (
    Copy,
    DayFile,
    Document,
    EmailHeaders,
    QuotedBlock,
    Rejected,
    Scan,
    SplitBody,
)
from rag_trend_radar.sources import enron

SOURCE = "enron"


def scan(root: Path) -> Scan:
    """Every file under `root`: the copies that parse, in path order, and the rejected ones."""
    copies: list[Copy] = []
    rejected: list[Rejected] = []
    with ProcessPoolExecutor() as pool:
        for scanned in pool.map(partial(_scan, root=root), enron.paths(root), chunksize=500):
            if isinstance(scanned, Rejected):
                rejected.append(scanned)
            else:
                copies.append(scanned)
    return Scan(
        mailboxes=tuple(sorted(p.name for p in root.iterdir() if p.is_dir())),
        copies=tuple(copies),
        rejected=tuple(rejected),
    )


def resolve(copies: Sequence[Copy]) -> dict[str, tuple[datetime, bool]]:
    """Each email's date over all its copies, and whether it may be hours off (by email key)."""
    dates: dict[str, list[datetime]] = defaultdict(list)
    for copy in copies:
        dates[copy.key].append(copy.sent_at)
    return {key: resolve_sent_at(group) for key, group in dates.items()}


def boilerplate(
    copies: Sequence[Copy], dates: dict[str, tuple[datetime, bool]], tz: ZoneInfo
) -> dict[str, date]:
    """Each boilerplate line and the local day it became one, counted over distinct emails."""
    own = {copy.key: copy.own for copy in reversed(copies)}  # the first copy by path wins
    dated = sorted((dates[key][0].astimezone(tz).date(), key) for key in own)
    return frequent_lines(((day, own[key]) for day, key in dated), BOILERPLATE_MIN_EMAILS)


def documents(
    files: Sequence[DayFile], held: Callable[[set[str]], set[str]]
) -> tuple[Document, ...]:
    """One day's documents: an email per distinct key, then each quoted message no email holds.

    `files` come sorted by (sent_at, key, path), so runs reproduce exactly. `held` takes text
    hashes and returns those that are some email's own text, in the whole dataset.
    """
    groups: dict[str, list[DayFile]] = defaultdict(list)
    for file in files:
        groups[file.key].append(file)
    emails = [_email(group) for group in groups.values()]
    quoted = {text_hash(b.text) for _, split in emails for b in split.quoted if b.text}
    return (*(doc for doc, _ in emails), *_orphans(emails, held(quoted)))


# Scanning, one file per worker call.


def _scan(path: Path, root: Path) -> Copy | Rejected:
    raw = enron.read_raw_file(root, path)
    try:
        email = enron.parse(raw)
    except ParseError as error:
        return Rejected(path=raw.path, reason=error.reason, detail=error.detail)
    return Copy(
        path=raw.path,
        key=content_hash(email.subject, email.body),
        sent_at=email.sent_at,
        own=split_quoted(normalize(email.body)).own,
    )


# Building documents.


def _email(group: list[DayFile]) -> tuple[Document, SplitBody]:
    first = group[0]
    email = enron.parse(first.raw)
    split = split_quoted(normalize(email.body))
    quoted = "\n\n".join(f"{b.header}\n\n{b.text}".strip() for b in split.quoted)
    document = Document(
        source=SOURCE,
        external_id=first.key,
        kind="email",
        sent_at=first.sent_at,
        title=email.subject,
        author=email.sender,
        body=split.own,
        quoted=quoted,
        public=False,
        readers=tuple(sorted({f.raw.mailbox for f in group})),
        paths=tuple(sorted(f.raw.path for f in group)),
        email=EmailHeaders.model_validate(email.model_dump(include=set(EmailHeaders.model_fields))),
        date_uncertain=first.date_uncertain,
    )
    return document, split


def _orphans(emails: list[tuple[Document, SplitBody]], held: set[str]) -> list[Document]:
    # Emails arrive in (sent_at, key) order, so a quoted message is dated by its first quote.
    first: dict[str, tuple[Document, QuotedBlock]] = {}
    quoted_by: dict[str, list[str]] = defaultdict(list)
    for doc, split in emails:
        for block in split.quoted:
            digest = text_hash(block.text)
            if not block.text or digest in held:
                continue
            first.setdefault(digest, (doc, block))
            if doc.external_id not in quoted_by[digest]:
                quoted_by[digest].append(doc.external_id)
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
            public=False,
            readers=(),
            quoted_by=tuple(quoted_by[digest]),
        )
        for digest, (doc, block) in first.items()
    ]
