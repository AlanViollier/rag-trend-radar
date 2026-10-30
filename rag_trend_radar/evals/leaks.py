"""Leak checks at full scale: no result a person can't read, none sent after the day asked."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import EvalQuestion, Record, SearchHit
from rag_trend_radar.search import hybrid, keyword, vector
from rag_trend_radar.search.ranking import cutoff


class Leak(Record):
    """One result that should never have come back."""

    kind: str
    method: str
    question_id: str
    person: str
    document_id: int


class Leaks(Record):
    """Results checked per method and kind, and every one that broke a rule."""

    checked: dict[str, int]
    leaks: tuple[Leak, ...]


def check(
    questions: Sequence[EvalQuestion], vectors: NDArray[np.float32], model: str, depth: int
) -> Leaks:
    """Search twice per question and check every result as the owner, outside row-level security.

    Permission: asked as a mailbox that can't read the labelled document, every result must be
    in that mailbox's `access` rows (the policy and the table agree). Time: asked as the question's
    person on the day before the labelled document was sent, every result must be sent before
    that day ended (each search applies the cutoff itself; nothing in the database enforces it).
    """
    checked: dict[str, int] = defaultdict(int)
    leaks: list[Leak] = []
    with db.connect() as owner:
        for q, vector_ in zip(questions, vectors, strict=True):
            label, sent = owner.execute(
                "SELECT id, (sent_at AT TIME ZONE %s)::date FROM documents WHERE external_id = %s",
                (get_settings().timezone, q.relevant[0]),
            ).fetchall()[0]
            outsider = owner.execute(
                "SELECT mailbox FROM employees e WHERE NOT EXISTS (SELECT 1 FROM access a"
                " WHERE a.document_id = %s AND a.mailbox = e.mailbox)"
                " ORDER BY md5(e.mailbox || %s) LIMIT 1",
                (label, q.id),
            ).fetchall()[0][0]
            before = sent - timedelta(days=1)
            for kind, person, day in (
                ("permission", outsider, q.as_of),
                ("time", q.person, before),
            ):
                for method, hits in _searches(q.question, vector_, model, person, day, depth):
                    ids = list({h.document_id for h in hits})
                    checked[f"{kind}, {method}"] += len(hits)
                    bad = (
                        unreadable(owner, person, ids)
                        if kind == "permission"
                        else too_late(owner, day, ids)
                    )
                    leaks += [
                        Leak(
                            kind=kind, method=method, question_id=q.id, person=person, document_id=d
                        )
                        for d in bad
                    ]
    return Leaks(checked=dict(checked), leaks=tuple(leaks))


def unreadable(
    owner: psycopg.Connection[TupleRow], person: str, document_ids: Sequence[int]
) -> list[int]:
    """The documents `person` has no `access` row for, read as the owner."""
    rows = owner.execute(
        "SELECT d FROM unnest(%s::bigint[]) d WHERE NOT EXISTS"
        " (SELECT 1 FROM access a WHERE a.document_id = d AND a.mailbox = %s) ORDER BY d",
        (list(document_ids), person),
    ).fetchall()
    return [d for (d,) in rows]


def too_late(
    owner: psycopg.Connection[TupleRow], day: date, document_ids: Sequence[int]
) -> list[int]:
    """The documents sent after `day` ended, read as the owner."""
    rows = owner.execute(
        "SELECT id FROM documents WHERE id = ANY(%s) AND sent_at >= %s ORDER BY id",
        (list(document_ids), cutoff(day)),
    ).fetchall()
    return [d for (d,) in rows]


def _searches(
    question: str, vector_: NDArray[np.float32], model: str, person: str, day: date, depth: int
) -> list[tuple[str, list[SearchHit]]]:
    with db.connect(person) as conn:
        return [
            ("keyword", keyword.search(conn, question, day, depth)),
            ("vector", vector.search(conn, vector_, model, day, depth)),
            ("hybrid", hybrid.search(conn, question, vector_, model, day, depth)),
        ]
