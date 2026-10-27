"""Write documents, chunks and vectors; carry deletes through to everything derived."""

from collections.abc import Callable, Sequence

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.ingest.dedup import text_hash
from rag_trend_radar.schema import Assembled, Document


def store(
    conn: psycopg.Connection[TupleRow],
    assembled: Assembled,
    pieces: Callable[[Document], list[str]],
) -> int:
    """Write a span: files as read, quarantine, employees, then documents and their chunks.

    Files already stored stay as they were (rule 4). Returns how many chunks were written.
    """
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO raw_files (path, mailbox, folder, content) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (path) DO NOTHING",
            [(f.path, f.mailbox, f.folder, f.content) for f in assembled.raw_files],
        )
        cur.executemany(
            "INSERT INTO quarantine (path, reason) VALUES (%s, %s) ON CONFLICT (path) DO NOTHING",
            list(assembled.rejected.items()),
        )
        cur.executemany(
            "INSERT INTO employees (mailbox) VALUES (%s) ON CONFLICT DO NOTHING",
            [(m,) for m in assembled.mailboxes],
        )
    return upsert_documents(conn, assembled.documents, pieces)


def upsert_documents(
    conn: psycopg.Connection[TupleRow],
    documents: Sequence[Document],
    pieces: Callable[[Document], list[str]],
) -> int:
    """Insert new documents and re-chunk changed ones; return how many chunks need embedding.

    Who may read a document is rewritten every time from `readers`, so a copy found in one more
    mailbox widens access without touching the chunks (rule 5: recounted, never incremented).
    """
    stored: dict[str, str] = dict(
        conn.execute(
            "SELECT external_id, body_hash FROM documents WHERE external_id = ANY(%s)",
            ([d.external_id for d in documents],),
        ).fetchall()
    )
    written = 0
    with conn.cursor() as cur:
        for doc in documents:
            document_id = _upsert_document(cur, doc)
            if doc.email is not None:
                e = doc.email
                cur.execute(
                    "INSERT INTO emails (document_id, message_id, sender, to_addrs, cc_addrs,"
                    " bcc_addrs, date_uncertain) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (document_id) DO UPDATE SET message_id = EXCLUDED.message_id,"
                    " sender = EXCLUDED.sender, to_addrs = EXCLUDED.to_addrs,"
                    " cc_addrs = EXCLUDED.cc_addrs, bcc_addrs = EXCLUDED.bcc_addrs,"
                    " date_uncertain = EXCLUDED.date_uncertain",
                    (
                        document_id,
                        e.message_id,
                        e.sender,
                        list(e.to_addrs),
                        list(e.cc_addrs),
                        list(e.bcc_addrs),
                        doc.date_uncertain,
                    ),
                )
            cur.execute("DELETE FROM access WHERE document_id = %s", (document_id,))
            cur.execute(
                "INSERT INTO access (document_id, mailbox) SELECT %s, unnest(%s::text[])",
                (document_id, list(doc.readers)),
            )
            if doc.paths:
                cur.execute(
                    "UPDATE raw_files SET document_id = %s WHERE path = ANY(%s)",
                    (document_id, list(doc.paths)),
                )
            if stored.get(doc.external_id) != doc.body_hash:
                written += _rechunk(cur, document_id, pieces(doc))
    return written


def write_vectors(
    conn: psycopg.Connection[TupleRow],
    model: str,
    chunk_ids: Sequence[int],
    vectors: NDArray[np.float32],
) -> None:
    """Store vectors in vectors_<model>, creating the table and its HNSW index on first use."""
    raise NotImplementedError


def delete_document(conn: psycopg.Connection[TupleRow], document_id: int) -> None:
    """Mark a document deleted and remove its chunks and vectors."""
    raise NotImplementedError


def _upsert_document(cur: psycopg.Cursor[TupleRow], doc: Document) -> int:
    cur.execute(
        "INSERT INTO documents (source, external_id, kind, sent_at, title, body, quoted,"
        " body_hash, public) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (source, external_id) DO UPDATE SET kind = EXCLUDED.kind,"
        " sent_at = EXCLUDED.sent_at, title = EXCLUDED.title, body = EXCLUDED.body,"
        " quoted = EXCLUDED.quoted, body_hash = EXCLUDED.body_hash, public = EXCLUDED.public"
        " RETURNING id",
        (
            doc.source,
            doc.external_id,
            doc.kind,
            doc.sent_at,
            doc.title,
            doc.body,
            doc.quoted,
            doc.body_hash,
            doc.public,
        ),
    )
    return int(cur.fetchall()[0][0])


def _rechunk(cur: psycopg.Cursor[TupleRow], document_id: int, texts: list[str]) -> int:
    ## Deleting the chunks takes their vectors with them (ON DELETE CASCADE).
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    cur.execute(
        "INSERT INTO chunks (document_id, position, text, text_hash)"
        " SELECT %s, n - 1, t, h FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY AS u(t, h, n)",
        (document_id, texts, [text_hash(t) for t in texts]),
    )
    return len(texts)
