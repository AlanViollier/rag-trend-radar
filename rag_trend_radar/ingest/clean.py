"""Clean an email body: strip markup, split own text from quoted messages, trim boilerplate."""

import html
import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from itertools import pairwise

from rag_trend_radar.schema import QuotedBlock, SplitBody

## Lines that open a quoted or forwarded message, as Outlook and Lotus Notes wrote them. Every
## pattern allows a leading `>` so a quote inside a quote still splits.
_Q = r"^[> \t]*"
_OUTLOOK = re.compile(_Q + r"-{2,}\s*Original Message\s*-{2,}[ \t]*$")
_LOTUS_FORWARD = re.compile(_Q + r"-{3,} ?Forwarded by ")
_OUTLOOK_HEADER = re.compile(
    _Q + r"From:.*\n" + _Q + r"(?:Sent|Date):.*\n" + _Q + r"To:", re.MULTILINE
)
_ON_WROTE = re.compile(_Q + r"On .{5,100}wrote:[ \t]*$")
## Lotus: `Nancy Hall@ENRON` then `10/02/2000 06:42 AM`, or both on one line; `To:` follows.
## Some add a zone: `on 10/30/2001 05:23 PM CST`.
_LOTUS_DATE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?(?:\s+[A-Z]{3})?[ \t]*$"
)
_BARE_DATE = re.compile(_Q + _LOTUS_DATE.pattern)
_SENT_BY = re.compile(_Q + r"Sent by:")
_TO = re.compile(_Q + r"To:")
_CC = re.compile(_Q + r"cc:", re.I)
_HEADER_FIELD = re.compile(
    _Q + r"(?:From|To|Sent|Date|cc|bcc|Subject|Importance|Sent by)\s*:", re.I
)
_SUBJECT = re.compile(_Q + r"Subject\s*:", re.I)
_FROM = re.compile(r"^From\s*:\s*", re.I)
## Where an author's name ends: a date, an address in brackets, or Lotus routing (`/HOU/ECT`).
_AUTHOR_END = re.compile(r"\s+on\s+\d|\s*\[mailto:|\s*<|/|\s+\d{1,2}/\d{1,2}/\d{2,4}")
## The rest of a "Forwarded by" line wrapped onto the next: `PM -----` or only the dashes.
_FORWARD_WRAP = re.compile(_Q + r"(?:[\d: ]*[AP]M )?-{3,}[ \t]*$")
_QUOTE_PREFIX = re.compile(r"^>+ ?", re.MULTILINE)

## Markup some mail clients left in the plain-text body. Style and script blocks go with their
## contents, and comments too: a browser shows none of it, and newsletters hide "your client
## doesn't support HTML" notes there.
_HIDDEN = re.compile(r"(?is)<(style|script)\b[^>]*>.*?</\1\s*>|<!--.*?-->|<!doctype\b[^>\n]*>")
_TAG_BREAK = re.compile(r"(?i)<(?:br|/p|/div|/tr|/li)\b[^>\n]*>")
_TAG = re.compile(
    r"(?i)</?(?:html|head|body|div|p|font|table|tbody|tr|td|th|span|b|i|u|strong|em|ul|ol|li|a"
    r"|img|meta|center|hr|title)\b[^>\n]*>"
)
_MIME = re.compile(
    r"^(?:This is a multi-part message in MIME format\.|-{2,}[=_\w.]*NextPart[^\n]*"
    r"|Content-(?:Type|Transfer-Encoding|Disposition|ID|Description):[^\n]*(?:\n[ \t]+[^\n]*)*)\n?",
    re.MULTILINE,
)
_BASE64_RUN = re.compile(
    r"^(?:[A-Za-z0-9+/]{60,}={0,2}\n){3,}(?:[A-Za-z0-9+/]*={0,2}\n)?", re.MULTILINE
)
## A trailing line found in at least this many distinct emails is boilerplate. Picked from the
## inspection report (2026-10-23): at 200 the lines are signatures, footers and disclaimers; at
## 100 they start to include form-letter sentences and newsletter text.
BOILERPLATE_MIN_EMAILS = 200
_SUBJECT_WITHIN = 40  # Lotus To: lists wrap over dozens of lines before the Subject
_LOTUS_TO_WITHIN = 5  # the memo layout leaves blank lines between From and To


def normalize(body: str) -> str:
    """Drop MIME parts, base64 runs, hidden HTML and tags, decode entities (`&amp;` → `&`)."""
    body = _BASE64_RUN.sub("", body)
    body = _MIME.sub("", body)
    body = _HIDDEN.sub("", body)
    body = _TAG_BREAK.sub("\n", body)
    return html.unescape(_TAG.sub("", body))


def split_quoted(body: str) -> SplitBody:
    """Separate the new text from quoted replies and forwards.

    Own text is everything above the first quoted message, minus `>` lines; each quoted message
    becomes a block of its header lines and its text, outermost first. Nothing is dropped but
    `>` marks and blank edges: `>` lines among the own text become a block with an empty header.
    """
    lines = body.split("\n")
    starts = _block_starts(lines)
    head = lines[: starts[0]] if starts else lines
    own = [line for line in head if not line.lstrip().startswith(">")]
    inline = [line for line in head if line.lstrip().startswith(">")]
    blocks = [QuotedBlock(header="", text=_unquote("\n".join(inline)))] if inline else []
    blocks += [_block(lines[start:end]) for start, end in pairwise([*starts, len(lines)])]
    return SplitBody(own="\n".join(own).strip(), quoted=tuple(blocks))


def frequent_lines(dated: Iterable[tuple[date, str]], min_emails: int) -> dict[str, date]:
    """Lines (spaces collapsed) found in at least `min_emails` texts, with the day they got there.

    `dated` holds one (day, text) per distinct email, in day order. A line is frequent as of a
    day from the day its `min_emails`th text was sent.
    """
    counts: Counter[str] = Counter()
    since: dict[str, date] = {}
    for day, text in dated:
        keys = {key for line in text.split("\n") if (key := _key(line))}
        counts.update(keys)
        since.update({key: day for key in keys if counts[key] == min_emails})
    return since


def strip_boilerplate(own: str, frequent: frozenset[str]) -> str:
    """Trim the trailing run of frequent lines (signature, disclaimer), never the whole text.

    Only the end is trimmed: a form letter whose every line repeats across thousands of senders
    is content, and it survives because trimming it would leave nothing.
    """
    lines = own.split("\n")
    end = len(lines)
    while end and (not lines[end - 1].strip() or _key(lines[end - 1]) in frequent):
        end -= 1
    return "\n".join(lines[:end]).rstrip() if end else own


def block_subject(header: str) -> str:
    """The Subject field of a quoted message's header, or "" when it has none."""
    for line in header.split("\n"):
        if match := _SUBJECT.match(line):
            return line[match.end() :].strip()
    return ""


def block_author(header: str) -> str:
    """Who wrote a quoted message, as its header names them, or "" when it doesn't.

    The first line that isn't a forward banner names the author in every layout that has one:
    `From: Mark Palmer on 07/05/2001`, Lotus `Robert C Williams/ENRON@enronXgate`,
    `"Keith McCrea" <KMCCREA@SABLAW.COM> on …`. Routing and dates are cut off.
    """
    for line in header.split("\n"):
        line = _unquote(line).strip()
        if not line or _LOTUS_FORWARD.match(line) or _FORWARD_WRAP.match(line):
            continue
        if _OUTLOOK.match(line) or _SENT_BY.match(line):
            continue
        if _HEADER_FIELD.match(line) and not _FROM.match(line):
            return ""
        name = _AUTHOR_END.split(_FROM.sub("", line), maxsplit=1)[0].strip(' "')
        return name.split("@")[0] if " " in name else name
    return ""


## Finding where quoted messages start.


def _block_starts(lines: list[str]) -> list[int]:
    starts = set()
    for i in range(len(lines)):
        if _opens_block(lines, i):
            starts.add(i)
        if (start := _lotus_header_start(lines, i)) is not None:
            starts.add(start)
    return _merge_adjacent(lines, sorted(starts))


def _opens_block(lines: list[str], i: int) -> bool:
    line = lines[i]
    return bool(
        _OUTLOOK.match(line)
        or _LOTUS_FORWARD.match(line)
        or _ON_WROTE.match(line)
        or _OUTLOOK_HEADER.match("\n".join(lines[i : i + 3]))
        or _dateless_lotus_header(lines, i)
    )


def _dateless_lotus_header(lines: list[str], i: int) -> bool:
    ## Some Lotus quotes keep only `To:`, `cc:` and `Subject:`: no sender line, no date.
    if not _TO.match(lines[i]):
        return False
    window = lines[i + 1 : i + _SUBJECT_WITHIN]
    subject = next((j for j, line in enumerate(window) if _SUBJECT.match(line)), None)
    return subject is not None and any(_CC.match(line) for line in window[:subject])


def _lotus_header_start(lines: list[str], i: int) -> int | None:
    ## A line ending in a Lotus date, with `To:` among the next two non-blank lines.
    following = [line for line in lines[i + 1 : i + 1 + _LOTUS_TO_WITHIN] if line.strip()][:2]
    if not _LOTUS_DATE.search(lines[i]) or not any(_TO.match(line) for line in following):
        return None
    start = i
    if _BARE_DATE.match(lines[i]):
        start = i - 1
        if start > 0 and _SENT_BY.match(lines[start]):
            start -= 1
    ## The memo layout indents its header with tabs, company name first: take those lines too.
    while start > 0 and lines[start - 1].startswith("\t"):
        start -= 1
    return max(start, 0)


def _merge_adjacent(lines: list[str], starts: list[int]) -> list[int]:
    ## "-----Original Message-----" then "From: … Sent: … To:" is one boundary, not two; so is a
    ## Lotus "Forwarded by" line followed by the forwarded message's own Lotus header. Each
    ## start is compared with the one just before it, kept or not, so a chain merges whole.
    merged = starts[:1]
    for previous, start in pairwise(starts):
        if not all(_is_header_line(line) for line in lines[previous + 1 : start]):
            merged.append(start)
    return merged


def _is_header_line(line: str) -> bool:
    return not line.strip() or bool(
        _HEADER_FIELD.match(line) or _FORWARD_WRAP.match(line) or _LOTUS_DATE.search(line)
    )


## Cutting one quoted message into header and text.


def _block(lines: list[str]) -> QuotedBlock:
    ## With a Subject, the header runs to the first blank line after it (Lotus To: lists wrap
    ## without indenting). Without one ("On … wrote:"), it's the opening line and any header
    ## fields right below it.
    subject = next(
        (i for i, line in enumerate(lines[:_SUBJECT_WITHIN]) if _SUBJECT.match(line)), None
    )
    if subject is not None:
        end = next((i for i in range(subject + 1, len(lines)) if not lines[i].strip()), len(lines))
    else:
        end = next((i for i in range(1, len(lines)) if not _is_header_line(lines[i])), len(lines))
    return QuotedBlock(
        header="\n".join(lines[:end]).strip(),
        text=_unquote("\n".join(lines[end:])).strip(),
    )


def _unquote(text: str) -> str:
    return _QUOTE_PREFIX.sub("", text)


def _key(line: str) -> str:
    return " ".join(line.split())
