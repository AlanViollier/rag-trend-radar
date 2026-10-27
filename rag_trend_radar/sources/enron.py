"""The Enron dataset (CMU, May 2015): one folder per mailbox, one file per email."""

import email
import email.utils
import re
from collections.abc import Iterator
from datetime import datetime
from email.header import decode_header
from email.message import Message
from email.policy import compat32
from pathlib import Path
from typing import cast

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.schema import ParsedEmail, RawFile

## 1997–2002 holds real mail. Outside it: the 1979-12-31 "no date" placeholder, a few 1986
## stragglers, and 2004+ stamps from when CMU processed the files.
FIRST_YEAR, LAST_YEAR = 1997, 2002

## Machine-written files with nothing to search: bounces paste back the failed message and its
## attachments, and <OMNI> files are address books sent to the Outlook migration.
_BOUNCE_SENDERS = ("postmaster@", "mailer-daemon@")
_BOUNCE_SUBJECTS = re.compile(
    r"(?i)^(?:undeliverable|returned mail|delivery status notification|failure notice"
    r"|mail delivery failed|delivery failure)"
)

_FOLD = re.compile(r"\r?\n[ \t]+")


def iter_raw_files(root: Path) -> Iterator[RawFile]:
    """Yield every email file under `root` (the extracted maildir/), in path order."""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        yield read_raw_file(root, path)


def read_raw_file(root: Path, path: Path) -> RawFile:
    """One file under `root`, its mailbox and folder taken from where it sits."""
    mailbox, *folders, _name = path.relative_to(root).parts
    return RawFile(
        path=path.relative_to(root).as_posix(),
        mailbox=mailbox,
        folder="/".join(folders),
        content=path.read_bytes(),
    )


def parse(raw: RawFile) -> ParsedEmail:
    """Read the headers and body of one file. Raises `ParseError` with the quarantine reason."""
    msg = email.message_from_bytes(raw.content, policy=compat32)
    sent_at = _sent_at(msg)
    if not FIRST_YEAR <= sent_at.year <= LAST_YEAR:
        raise ParseError(f"date outside {FIRST_YEAR}–{LAST_YEAR}: {msg['Date']}")
    sender, subject, body = email.utils.parseaddr(msg["From"])[1].lower(), _subject(msg), _body(msg)
    if sender.startswith(_BOUNCE_SENDERS) or _BOUNCE_SUBJECTS.match(subject):
        raise ParseError("bounce")
    if body.lstrip().startswith("<OMNI>"):
        raise ParseError("address book dump")
    return ParsedEmail(
        message_id=msg["Message-ID"].strip(),
        sent_at=sent_at,
        sender=sender,
        to_addrs=_addresses(msg, "To"),
        cc_addrs=_addresses(msg, "Cc"),
        bcc_addrs=_addresses(msg, "Bcc"),
        subject=subject,
        body=body,
    )


def _sent_at(msg: Message) -> datetime:
    try:
        return email.utils.parsedate_to_datetime(msg["Date"])
    except (TypeError, ValueError) as error:
        raise ParseError(f"unparseable date: {msg['Date']}") from error


def _addresses(msg: Message, header: str) -> tuple[str, ...]:
    pairs = email.utils.getaddresses(msg.get_all(header, []))
    return tuple(dict.fromkeys(address.lower() for _name, address in pairs if address))


def _subject(msg: Message) -> str:
    ## Encoded words (`=?cp1252?Q?...?=`) come back as bytes, sometimes with no charset at all.
    ## A long subject is folded over lines (a line break, then a space or tab); compat32 keeps
    ## the fold, so it's unfolded here.
    parts = decode_header(_FOLD.sub(" ", msg["Subject"] or ""))
    return "".join(
        part if isinstance(part, str) else _text(part, charset) for part, charset in parts
    )


def _body(msg: Message) -> str:
    ## Every CMU file is single-part text/plain, so the payload is bytes. decode=True undoes
    ## quoted-printable (`=20`, soft `=` line breaks).
    payload = cast(bytes, msg.get_payload(decode=True))
    return _text(payload, msg.get_content_charset()).replace("\r\n", "\n")


def _text(data: bytes, charset: str | None) -> str:
    try:
        return data.decode(charset or "us-ascii")
    except UnicodeDecodeError, LookupError:
        ## Mislabelled mail is mostly Windows-1252 (Outlook's curly quotes, dashes).
        return data.decode("cp1252", errors="replace")
