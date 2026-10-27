"""Cut an email's text into pieces the embedding model takes whole, each led by its header."""

from collections.abc import Callable
from datetime import datetime
from typing import cast

from semchunk.semchunk import chunk as split

from rag_trend_radar.ingest.clean import strip_boilerplate
from rag_trend_radar.schema import Document

## 90% of own texts are under ~400 tokens, so most emails stay whole. 256 and 1,024 are rival
## rows in the M1 recall table. No overlap between pieces: semchunk's overlap adds a piece that
## straddles the paragraph break, mixing the two topics the split had just separated.
MAX_TOKENS = 512


def pieces(
    document: Document, count_tokens: Callable[[str], int], boilerplate: frozenset[str]
) -> list[str]:
    """The chunks of a document: its own text minus trailing boilerplate, led by its header."""
    sent_at = document.sent_at if document.kind == "email" else None
    lead = header(document.title, document.author, sent_at)
    return chunk(strip_boilerplate(document.body, boilerplate), count_tokens, lead=lead)


def header(subject: str, sender: str = "", sent_at: datetime | None = None) -> str:
    """The lines that lead every chunk: a bare "sounds good" is findable by its subject.

    A field that isn't known is left out rather than guessed.
    """
    lines = [f"Subject: {subject}"]
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
    ## semchunk returns (pieces, spans) when asked for offsets; its annotation doesn't say so.
    _pieces, spans = cast(
        "tuple[list[str], list[tuple[int, int]]]",
        split(text, budget, count_tokens, offsets=True),
    )
    return [prefix + text[start:end] for start, end in spans]
