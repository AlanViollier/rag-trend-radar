"""What a short reply answered: without it, "Yes, approved" tells a reader nothing."""

from collections.abc import Sequence

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar.schema import SearchHit

## A reply this short rarely says what it's about. 58 of 2,080 result slots are, against a
## median 1,037 characters of quote (2026-11-02), so this costs few tokens.
SHORT_BODY = 200
## The messages above an email come outermost first, so the one it answered is the start.
QUOTE_CHARS = 1500


def with_quotes(conn: psycopg.Connection[TupleRow], hits: Sequence[SearchHit]) -> list[SearchHit]:
    """`hits` with the start of what each short reply quotes, read as the person searching.

    An email's quote sits on its own row: whoever can read the email can read it, and it was
    written before the email, so neither permissions nor `as_of` need a further check.
    """
    rows: dict[int, str] = dict(
        conn.execute(
            "SELECT id, quoted FROM documents WHERE id = ANY(%s) AND length(body) < %s"
            " AND quoted <> ''",
            ([hit.document_id for hit in hits], SHORT_BODY),
        ).fetchall()
    )
    return [
        hit.model_copy(update={"context": rows[hit.document_id][:QUOTE_CHARS]})
        if hit.document_id in rows
        else hit
        for hit in hits
    ]
