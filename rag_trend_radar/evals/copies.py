"""Copies of a labelled email count as right answers: decided by comparing text, never by a model.

A labelled question has one right document by construction, but the full store holds copies of
it the dedup can't merge: a forward or reply carrying its text, the same notice re-sent with
another header, a deal number repeated in the next email about that deal. A strict score counts
those as misses.
"""

from collections.abc import Iterable, Mapping, Sequence

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.ingest.dedup import numbers, words
from rag_trend_radar.schema import EvalQuestion

# Shared words over all words (Jaccard) from which two texts are the same message re-sent.
# The judge's verdicts (2026-10-30) put real re-sends well above it and look-alikes below.
SHARED = 0.5
# A label shorter than this ("Thanks, see you there") is contained in too many emails to count.
MIN_CONTAINED = 100


def copies(
    conn: psycopg.Connection[TupleRow],
    questions: Sequence[EvalQuestion],
    ranked: Iterable[Mapping[str, Sequence[str]]],
) -> dict[str, set[str]]:
    """Each question's labels, plus every ranked document that is a copy of one of them.

    `ranked` is any number of {question id: external ids} rankings; every document they hold is
    checked. A copy holds a label's own text (quoted or as its own), or shares `SHARED` of its
    words and every one of its numbers.
    """
    wanted: dict[str, set[str]] = {q.id: set() for q in questions}
    for ranking in ranked:
        for q in questions:
            wanted[q.id] |= set(ranking.get(q.id, ()))
    texts = _texts(conn, {x for ids in wanted.values() for x in ids} | _labels(questions))
    found: dict[str, set[str]] = {}
    for q in questions:
        labels = [texts[x] for x in q.relevant]
        found[q.id] = set(q.relevant) | {
            x
            for x in wanted[q.id] - set(q.relevant)
            if any(is_copy(label, texts[x]) for label in labels)
        }
    return found


def is_copy(label: tuple[str, str], other: tuple[str, str]) -> bool:
    """Whether `other` (own text, quoted text) carries `label`'s message.

    A codes question gets no rule of its own: another email naming the same deal needn't hold
    the fact asked about (counting them added 123 "answers" and lifted codes from .909 to .990).
    """
    own, _ = label
    flat = _flat(own)
    if len(flat) >= MIN_CONTAINED and flat in _flat(" ".join(other)):
        return True
    # Shared words alone take templates for copies (NYISO notices, calendar entries, one
    # disclaimer under different notes); the part that differs is the numbers, so a re-send
    # must carry every number the label does.
    a, b = words(own), words(other[0])
    shared = bool(a and b) and len(a & b) / len(a | b) >= SHARED
    return shared and numbers(own) <= numbers(other[0])


def _texts(
    conn: psycopg.Connection[TupleRow], external_ids: set[str]
) -> dict[str, tuple[str, str]]:
    rows = conn.execute(
        "SELECT external_id, body, quoted FROM documents WHERE external_id = ANY(%s)",
        (list(external_ids),),
    ).fetchall()
    return {x: (body, quoted) for x, body, quoted in rows}


def _labels(questions: Sequence[EvalQuestion]) -> set[str]:
    return {x for q in questions for x in q.relevant}


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()
