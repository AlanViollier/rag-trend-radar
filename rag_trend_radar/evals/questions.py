"""Labelled questions: a real email, and a question it answers asked in other words."""

import hashlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from enum import StrEnum
from pathlib import Path

import anthropic
import psycopg
from psycopg.rows import TupleRow
from pydantic import BaseModel

from rag_trend_radar.config import get_settings
from rag_trend_radar.schema import EvalQuestion, Record

## Bump on any change to the prompt or the sample: cached questions are keyed by it.
PROMPT_VERSION = 3
## Too short and there's no fact to ask about; too long and many facts compete, so a question
## about one of them is answered as well by other emails in the same digest.
MIN_CHARS, MAX_CHARS = 200, 4000
SYSTEM = """You write test questions for a search engine over Enron's email from 2001. Each \
question checks whether the engine finds one email, so it must be answerable from that email \
and, as far as you can tell, from few others."""
INSTRUCTIONS = """Here is an email from {mailbox}'s mailbox.

<email>
Subject: {title}
From: {author}
Date: {day}

{body}
</email>

Write one question {mailbox} might type to find what this email says. Rules:
- Ask about one specific fact the email states: who, what, how much, when, or what was decided. \
One fact, one question. Never ask about something the email doesn't say.
- {words_rule}
- Don't mention "this email" or "the email": the person asking hasn't seen it.
- If the email holds no fact worth asking about, set usable to false: bounce or delivery-failure \
notices, other automated messages, empty or pleasantry-only mail."""


class Style(StrEnum):
    """What a question may borrow from its email: the three sets test different searches."""

    ## Names allowed: the everyday question.
    NAMES = "names"
    ## Meaning over matching: without exact names, keyword search loses its edge, and what's
    ## left is how well vectors capture what a question means.
    NO_NAMES = "no-names"
    ## One identifier quoted as written: the case vectors blur ("617161" near any number) and
    ## keyword search is there for.
    CODES = "codes"


_OWN_WORDS = (
    "Use your own words. Don't copy the subject line, numbers as written, distinctive phrases, "
    "or rare terms; "
)
WORDS_RULES = {
    Style.NAMES: _OWN_WORDS + "a person or company may be named when the question needs it.",
    Style.NO_NAMES: _OWN_WORDS
    + 'name no person, company, place or product: describe them instead ("the Canadian '
    'pipeline company", "our head of research").',
    Style.CODES: "Quote exactly one identifier from the email as written (a deal number, contract, "
    "ticket or invoice number) the way someone looking for that record would type it; put "
    "everything else in your own words. Phone numbers, zip codes, dates and amounts aren't "
    "identifiers: if the email holds none, set usable to false.",
}
## A code-style source must hold a run of five or more digits; Claude sorts out the phone
## numbers and zip codes.
CODE_PATTERN = "[0-9]{5,}"


class Source(Record):
    """An email or quoted message chosen to ask about."""

    external_id: str
    kind: str
    title: str
    author: str
    day: date
    body: str
    mailbox: str


class _Written(BaseModel):
    usable: bool
    question: str


def sample(
    conn: psycopg.Connection[TupleRow], kind: str, n: int, seed: int, *, style: Style = Style.NAMES
) -> list[Source]:
    """`n` documents of `kind` with chunks and a body of usable length, in a seeded random order.

    For the codes style, only documents holding a long digit run are drawn.

    The mailbox asking is one that can read the document, picked by the same hash, so the
    question is asked as someone row-level security lets see the answer.
    """
    rows = conn.execute(
        "SELECT d.external_id, d.kind, d.title, e.sender, d.sent_at::date, d.body,"
        " (array_agg(a.mailbox ORDER BY md5(a.mailbox || %(seed)s)))[1]"
        " FROM documents d JOIN access a ON a.document_id = d.id"
        " LEFT JOIN emails e ON e.document_id = d.id"
        " WHERE d.kind = %(kind)s AND length(d.body) BETWEEN %(lo)s AND %(hi)s"
        " AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"
        " AND (%(pattern)s::text IS NULL OR d.body ~ %(pattern)s)"
        " GROUP BY d.id, e.sender ORDER BY md5(d.external_id || %(seed)s) LIMIT %(n)s",
        {
            "kind": kind,
            "lo": MIN_CHARS,
            "hi": MAX_CHARS,
            "seed": str(seed),
            "n": n,
            "pattern": CODE_PATTERN if style is Style.CODES else None,
        },
    ).fetchall()
    return [
        Source(external_id=x, kind=k, title=t, author=a or "", day=d, body=b, mailbox=m)
        for x, k, t, a, d, b, m in rows
    ]


def write(
    sources: list[Source], as_of: date, model: str, *, style: Style = Style.NAMES
) -> Iterator[EvalQuestion]:
    """A question per usable source; unusable ones are skipped. Cached per source (rule 6)."""
    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())
    ## Calls run 8 at a time and come back in source order, so the set reproduces. Every source
    ## is asked, even past the ones the caller keeps: the spares are cached for the next run.
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(
            pool.map(lambda s: _cached(s, model, style) or _ask(client, s, model, style), sources)
        )
    for source, written in zip(sources, answers, strict=True):
        if written.usable and _one_question(written.question):
            yield EvalQuestion(
                id=f"q-{source.external_id[:12]}",
                question=written.question,
                as_of=as_of,
                person=source.mailbox,
                relevant=(source.external_id,),
            )


def _one_question(text: str) -> bool:
    ## 4 of the first 200 broke "one fact, one question": two questions in one, or a rewrite
    ## after "Actually, simplified:". A spare source takes their place.
    text = text.strip()
    return text.endswith("?") and text.count("?") == 1 and "\n" not in text and "<" not in text


def _ask(client: anthropic.Anthropic, source: Source, model: str, style: Style) -> _Written:
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": INSTRUCTIONS.format(
                    **source.model_dump(), words_rule=WORDS_RULES[style]
                ),
            }
        ],
        output_format=_Written,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    written = response.parsed_output
    if written is None:
        return _Written(usable=False, question="")
    path = _cache_path(source, model, style)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(written.model_dump_json())
    return written


def _cached(source: Source, model: str, style: Style) -> _Written | None:
    path = _cache_path(source, model, style)
    return _Written.model_validate_json(path.read_text()) if path.exists() else None


def _cache_path(source: Source, model: str, style: Style) -> Path:
    key = f"{PROMPT_VERSION}\x00{model}\x00{style}\x00{source.model_dump_json()}"
    digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
    return get_settings().data_dir / "cache" / "questions" / f"{digest}.json"
