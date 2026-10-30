"""The judge: Claude reads a question and a document and says whether it answers; code checks it.

A labelled question has one right document by construction, but a bigger store holds others that
answer as well (a forward, a near-identical notice), and a strict score counts those as misses.
The judge reads every document a method ranked above the label, and the label itself, which
catches questions whose own document no longer answers them.
"""

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path

import anthropic
import psycopg
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from psycopg.rows import TupleRow
from pydantic import BaseModel

from rag_trend_radar.config import get_settings
from rag_trend_radar.schema import EvalQuestion, Judgment, RecallRun, Record, Verdict

## Bump on any change to the prompt: cached verdicts are keyed by it.
PROMPT_VERSION = 1
MODEL = "claude-opus-5-5"
## The labelled document must rank inside this for recall@10 to count it; a document ranked
## above it there can take its place, so those are the ones the judge reads.
POOL = 10
## Batch API prices for MODEL, per million tokens (half the standard $4 / $20).
INPUT_PER_M, OUTPUT_PER_M = 2.0, 10.0
SYSTEM = """You check answers for a search engine over Enron's email from 2001. Given a question \
someone typed and one document, you decide whether that document answers it. Judge only from the \
document's text, never from what you know."""
INSTRUCTIONS = """{mailbox} typed this question into the search engine over their own mailbox; \
"I", "me" and "we" in it mean {mailbox} and their team.

<question>{question}</question>

<document>
{document}
</document>

Does this document answer the question?
- answers: it states the fact asked for. Someone reading only this document would have the \
answer, even if the wording differs from the question's.
- partly: it is about the same matter but leaves out the fact asked for, or gives only part of it.
- no: it doesn't.
For answers and partly, copy into quote the sentence or header line that states it, exactly as \
written in the document. For no, leave quote empty."""
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "quote": {"type": "string"},
    },
    "required": ["verdict", "quote"],
    "additionalProperties": False,
}


class Pair(Record):
    """A question and one document to read against it."""

    question: EvalQuestion
    external_id: str
    document: str


class Estimate(Record):
    """What a judge run would send, before it's sent."""

    pairs: int
    cached: int
    input_tokens: int
    dollars: float


class _Read(BaseModel):
    verdict: Verdict
    quote: str


def pairs(
    conn: psycopg.Connection[TupleRow], questions: Sequence[EvalQuestion], run: RecallRun | None
) -> list[Pair]:
    """Every (question, document) pair `pool` picks, with the document's text."""
    wanted = pool(questions, run)
    texts = documents(conn, {x for ids in wanted.values() for x in ids})
    return [
        Pair(question=q, external_id=x, document=texts[x])
        for q in questions
        for x in sorted(wanted[q.id])
    ]


def pool(questions: Sequence[EvalQuestion], run: RecallRun | None) -> dict[str, set[str]]:
    """Each question's own documents, and every document a method of `run` ranked above them.

    Within the top `POOL`; when the label isn't there, the whole top `POOL`. Pooled over every
    method, so no method is judged more kindly than another.
    """
    wanted: dict[str, set[str]] = {q.id: set(q.relevant) for q in questions}
    for ranked in (run.rankings if run else {}).values():
        for q in questions:
            top = ranked.get(q.id, [])[:POOL]
            label = min((top.index(x) for x in q.relevant if x in top), default=POOL)
            wanted[q.id] |= set(top[:label])
    return wanted


def documents(conn: psycopg.Connection[TupleRow], external_ids: Iterable[str]) -> dict[str, str]:
    """Each document as the judge reads it: subject, sender and date, then its own text."""
    rows = conn.execute(
        "SELECT d.external_id, d.title, e.sender, d.sent_at::date, d.body FROM documents d"
        " LEFT JOIN emails e ON e.document_id = d.id WHERE d.external_id = ANY(%s)",
        (list(external_ids),),
    ).fetchall()
    return {x: _render(t, a or "", day, b) for x, t, a, day, b in rows}


def estimate(todo: Sequence[Pair]) -> Estimate:
    """Tokens and dollars for the pairs not cached yet, at about 3.5 characters a token."""
    fresh = [p for p in todo if not _cache_path(p).exists()]
    chars = sum(
        len(SYSTEM) + len(INSTRUCTIONS) + len(p.question.question) + len(p.document) for p in fresh
    )
    tokens = int(chars / 3.5)
    ## The verdict JSON plus a little thinking: 81 output tokens a pair on the smoke batch
    ## (2026-10-30); 100 leaves room.
    dollars = tokens / 1e6 * INPUT_PER_M + len(fresh) * 100 / 1e6 * OUTPUT_PER_M
    return Estimate(
        pairs=len(todo), cached=len(todo) - len(fresh), input_tokens=tokens, dollars=dollars
    )


def judge(todo: Sequence[Pair], progress: Callable[[str], None]) -> list[Judgment]:
    """A judgment per pair: cached ones read back, the rest sent as one batch and waited for.

    A batch sent but not yet collected (the run was stopped) is collected first, never re-sent.
    Pairs the batch couldn't answer (errors, refusals) get no judgment; the next run retries them.
    `progress` hears the batch's state every minute.
    """
    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())
    pending = _cache_dir() / "pending-batch.txt"
    if pending.exists():
        _collect(client, pending.read_text().strip(), todo, progress)
        pending.unlink()
    fresh = {_key(p): p for p in todo if not _cache_path(p).exists()}
    if fresh:
        batch = client.messages.batches.create(
            requests=[Request(custom_id=key, params=_params(p)) for key, p in fresh.items()]
        )
        pending.write_text(batch.id)
        _collect(client, batch.id, todo, progress)
        pending.unlink()
    return [j for p in todo if (j := _cached(p)) is not None]


def quoted(quote: str, document: str) -> bool:
    """The quote appears in the document as written, up to whitespace and case."""
    flat = " ".join(quote.split()).casefold()
    return bool(flat) and flat in " ".join(document.split()).casefold()


def _collect(
    client: anthropic.Anthropic,
    batch_id: str,
    todo: Sequence[Pair],
    progress: Callable[[str], None],
) -> None:
    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        counts = batch.request_counts
        progress(
            f"batch {batch_id}: {counts.succeeded + counts.errored} done, {counts.processing} to go"
        )
        time.sleep(60)
    by_key = {_key(p): p for p in todo}
    for result in client.messages.batches.results(batch_id):
        pair = by_key.get(result.custom_id)
        ## Errors, expiries and refusals leave no cache file: the next run sends the pair again.
        if pair is None or result.result.type != "succeeded":
            continue
        message = result.result.message
        if message.stop_reason == "refusal":
            continue
        text = next(b.text for b in message.content if b.type == "text")
        _cache_path(pair).write_text(_Read.model_validate_json(text).model_dump_json())


def _params(pair: Pair) -> MessageCreateParamsNonStreaming:
    prompt = INSTRUCTIONS.format(
        mailbox=pair.question.person, question=pair.question.question, document=pair.document
    )
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
    )


def _cached(pair: Pair) -> Judgment | None:
    path = _cache_path(pair)
    if not path.exists():
        return None
    read = _Read.model_validate_json(path.read_text())
    return Judgment(
        question_id=pair.question.id,
        external_id=pair.external_id,
        verdict=read.verdict,
        quote=read.quote,
        quoted=quoted(read.quote, pair.document),
    )


def _key(pair: Pair) -> str:
    ## Everything the verdict depends on: the prompt, the model, who asks what, and the text read.
    q = pair.question
    key = f"{PROMPT_VERSION}\x00{MODEL}\x00{q.person}\x00{q.question}\x00{pair.document}"
    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()


def _cache_path(pair: Pair) -> Path:
    return _cache_dir() / f"{_key(pair)}.json"


def _cache_dir() -> Path:
    path = get_settings().data_dir / "cache" / "judge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _render(title: str, author: str, day: date, body: str) -> str:
    head = [
        f"Subject: {title}" if title else "",
        f"From: {author}" if author else "",
        f"Date: {day}",
    ]
    return "\n".join(line for line in head if line) + "\n\n" + body
