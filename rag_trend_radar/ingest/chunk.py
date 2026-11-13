"""Cut an email's text into pieces the embedding model takes whole, each led by its header."""

from collections.abc import Callable
from datetime import datetime
from typing import cast

from semchunk.semchunk import chunk as split

from rag_trend_radar.ingest.clean import split_quoted, strip_boilerplate
from rag_trend_radar.records import Document

# 90% of own texts are under ~400 tokens, so most emails stay whole. 256 and 1,024 scored within
# noise of it on the September questions. No overlap between pieces: semchunk's overlap adds a piece that
# straddles the paragraph break, mixing the two topics the split had just separated.
MAX_TOKENS = 512
# A reply this short says little alone: its vector is mostly its subject, and "i sent it today"
# can't be found by the names in the message it answers. 22 questions from a sample of such
# replies, investigator top 10: 14 → 20 with the quote searched too, other question sets flat
# (2026-11-03). Longer mail keeps its own text alone, so a thread doesn't repeat itself.
SHORT_REPLY = 200
QUOTE_CHARS = 600


def pieces(
    document: Document,
    count_tokens: Callable[[str], int],
    boilerplate: frozenset[str],
    *,
    max_tokens: int = MAX_TOKENS,
    with_header: bool = True,
) -> list[str]:
    """The chunks of a document: its own text minus trailing boilerplate, led by its header.

    `max_tokens` and `with_header` are the recall table's rival rows (256 and 1,024; bare body).
    A short reply also carries the start of what it answered, unless that would pass `max_tokens`.
    """
    sent_at = document.sent_at if document.kind == "email" else None
    lead = header(document.title, document.author, sent_at) if with_header else ""
    body = strip_boilerplate(document.body, boilerplate)
    texts = chunk(body, count_tokens, lead=lead, max_tokens=max_tokens)
    quote = reply_quote(document)
    if quote and len(texts) == 1 and count_tokens(f"{texts[0]}\n\n{quote}") <= max_tokens:
        return [f"{texts[0]}\n\n{quote}"]
    return texts


def reply_quote(document: Document) -> str:
    """The start of the message a short reply answered, or "" for any other document."""
    if document.kind != "email" or len(document.body) >= SHORT_REPLY or not document.quoted:
        return ""
    blocks = (block.text.strip() for block in split_quoted(document.quoted).quoted)
    return next((text[:QUOTE_CHARS] for text in blocks if text), "")


def header(subject: str, sender: str = "", sent_at: datetime | None = None) -> str:
    """The lines that lead every chunk: a bare "sounds good" is findable by its subject.

    A field that isn't known is left out rather than guessed, an empty subject too (8,784 emails).
    """
    lines = [f"Subject: {subject}"] if subject.strip() else []
    if sender:
        lines.append(f"From: {sender}")
    if sent_at is not None:
        lines.append(f"Date: {sent_at:%Y-%m-%d}")
    return "\n".join(lines)


def chunk(
    text: str, count_tokens: Callable[[str], int], *, lead: str = "", max_tokens: int = MAX_TOKENS
) -> list[str]:
    """`text` whole when it fits, else split at paragraphs, then sentences, then words.

    Every piece starts with `lead` and stays within `max_tokens` counted with it. Sizes are in
    the embedding model's own tokens, so `count_tokens` comes from its tokenizer.
    """
    if not text.strip():
        return []
    prefix = f"{lead}\n\n" if lead else ""
    budget = max_tokens - count_tokens(prefix)
    if count_tokens(text) <= budget:
        return [prefix + text]
    # semchunk returns (pieces, spans) when asked for offsets; its annotation doesn't say so.
    _pieces, spans = cast(
        "tuple[list[str], list[tuple[int, int]]]",
        split(text, budget, count_tokens, offsets=True),
    )
    return [prefix + text[start:end] for start, end in spans]
