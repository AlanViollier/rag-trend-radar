"""Write documents, chunks and vectors; carry deletes through to everything derived."""

import re
from collections.abc import Callable, Sequence

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings
from rag_trend_radar.ingest.dedup import text_hash
from rag_trend_radar.schema import Document, EmailHeaders


def upsert_documents(
    conn: psycopg.Connection[TupleRow],
    documents: Sequence[Document],
    pieces: Callable[[Document], list[str]],
) -> int:
    """Insert new documents and re-chunk changed ones; return how many chunks need embedding.

    Who may read an email is rewritten every time from `readers`, so a copy found in one more
    mailbox widens access without touching the chunks. A quoted message's readers are recounted
    from every stored email quoting it, whichever day quoted it (rule 5).
    """
    stored: dict[str, str] = dict(
        conn.execute(
            "SELECT external_id, body_hash FROM documents WHERE external_id = ANY(%s)",
            ([d.external_id for d in documents],),
        ).fetchall()
    )
    ids: dict[str, int] = {}
    written = 0
    with conn.cursor() as cur:
        for doc in documents:
            document_id = ids[doc.external_id] = _upsert_document(cur, doc)
            if doc.email is not None:
                _upsert_email(cur, document_id, doc.email, uncertain=doc.date_uncertain)
            if doc.kind != "quoted":
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
        _recount_quoted(cur, documents, ids)
    return written


def vectors_table(model: str, dims: int) -> str:
    """`vectors_embeddinggemma_2_768`: one table per model and size, so rivals sit side by side."""
    return f"vectors_{re.sub(r'\W+', '_', model.rsplit('/', 1)[-1].lower())}_{dims}"


def vectors_identifier(model: str, dims: int) -> sql.Identifier:
    """The vectors table in the configured schema, never another found through the search path.

    In an experiment's schema an unqualified name falls through to public when the experiment
    has no such table yet, and reads or writes the store's vectors (2026-10-27).
    """
    return sql.Identifier(get_settings().db_schema, vectors_table(model, dims))


def unembedded(conn: psycopg.Connection[TupleRow], model: str, dims: int) -> dict[str, list[int]]:
    """Chunks with no vector yet in `model` at `dims`, grouped by text: equal texts embed once."""
    table = ensure_vectors(conn, model, dims)
    rows = conn.execute(
        sql.SQL(
            "SELECT text, array_agg(id ORDER BY id) FROM chunks"
            " WHERE NOT EXISTS (SELECT 1 FROM {} v WHERE v.chunk_id = chunks.id)"
            " GROUP BY text ORDER BY min(id)"
        ).format(table)
    ).fetchall()
    return dict(rows)


def reuse_vectors(conn: psycopg.Connection[TupleRow], model: str, dims: int, source: str) -> int:
    """Copy `source` schema's vector for every chunk here whose text it already embedded.

    A vector depends only on the text, so an experiment that re-chunks only the long emails
    re-embeds only those: the rest match the store's chunks by text hash. Returns how many.
    """
    table = ensure_vectors(conn, model, dims)
    cur = conn.execute(
        sql.SQL(
            "INSERT INTO {} (chunk_id, embedding)"
            " SELECT DISTINCT ON (c.id) c.id, v.embedding FROM chunks c"
            " JOIN {} s ON s.text_hash = c.text_hash JOIN {} v ON v.chunk_id = s.id"
            " WHERE NOT EXISTS (SELECT 1 FROM {} x WHERE x.chunk_id = c.id)"
            " ORDER BY c.id ON CONFLICT (chunk_id) DO NOTHING"
        ).format(
            table,
            sql.Identifier(source, "chunks"),
            sql.Identifier(source, vectors_table(model, dims)),
            table,
        )
    )
    return cur.rowcount


def derive_vectors(conn: psycopg.Connection[TupleRow], model: str, dims: int, wider: int) -> int:
    """Fill `model` at `dims` from its `wider` vectors: the first `dims` numbers, renormalised.

    Only for models trained with Matryoshka representation learning (EmbeddingGemma), whose
    leading dimensions carry a usable embedding alone; the result equals embedding again with
    truncate_dim, without the 27 minutes. Returns how many vectors were added.
    """
    table = ensure_vectors(conn, model, dims)
    source = vectors_identifier(model, wider)
    cur = conn.execute(
        sql.SQL(
            "INSERT INTO {} (chunk_id, embedding)"
            " SELECT chunk_id, l2_normalize(subvector(embedding, 1, %s)) FROM {}"
            " ON CONFLICT (chunk_id) DO NOTHING"
        ).format(table, source),
        (dims,),
    )
    return cur.rowcount


def write_vectors(
    conn: psycopg.Connection[TupleRow],
    model: str,
    chunk_ids: Sequence[int],
    vectors: NDArray[np.float32],
) -> None:
    """Store vectors in vectors_<model>_<dims>, creating the table on first use."""
    table = ensure_vectors(conn, model, vectors.shape[1])
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "INSERT INTO {} (chunk_id, embedding) VALUES (%s, %s)"
                " ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ).format(table),
            [(i, HalfVector(v)) for i, v in zip(chunk_ids, vectors, strict=True)],
        )


def ensure_vectors(conn: psycopg.Connection[TupleRow], model: str, dims: int) -> sql.Identifier:
    """The vectors table of `model` at `dims`, created with its index and policy on first use."""
    ## Readers see a vector only when they see its chunk; the grant is by name like every table.
    table = vectors_identifier(model, dims)
    exists = conn.execute("SELECT to_regclass(%s)", (table.as_string(conn),)).fetchall()[0][0]
    if exists is None:
        for statement in (
            "CREATE TABLE {t} (chunk_id bigint PRIMARY KEY REFERENCES chunks (id)"
            " ON DELETE CASCADE, embedding halfvec({dims}) NOT NULL)",
            "CREATE INDEX ON {t} USING hnsw (embedding halfvec_cosine_ops)",
            "GRANT SELECT ON {t} TO radar_reader",
            "ALTER TABLE {t} ENABLE ROW LEVEL SECURITY",
            "CREATE POLICY visible ON {t} FOR SELECT TO radar_reader"
            " USING (EXISTS (SELECT 1 FROM chunks WHERE chunks.id = {t}.chunk_id))",
        ):
            conn.execute(sql.SQL(statement).format(t=table, dims=sql.Literal(dims)))
    return table


def delete_document(conn: psycopg.Connection[TupleRow], document_id: int) -> None:
    """Mark a document deleted and remove its chunks and vectors."""
    raise NotImplementedError


def _upsert_document(cur: psycopg.Cursor[TupleRow], doc: Document) -> int:
    ## A quoted message comes back each day someone quotes it: it keeps the date and subject of
    ## its first quote. An email's date is resolved once, so for it LEAST changes nothing.
    cur.execute(
        "INSERT INTO documents (source, external_id, kind, sent_at, title, body, quoted,"
        " body_hash, public) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (source, external_id) DO UPDATE SET kind = EXCLUDED.kind,"
        " sent_at = LEAST(documents.sent_at, EXCLUDED.sent_at), title = CASE"
        " WHEN EXCLUDED.sent_at < documents.sent_at THEN EXCLUDED.title ELSE documents.title END,"
        " body = EXCLUDED.body,"
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


def _upsert_email(
    cur: psycopg.Cursor[TupleRow], document_id: int, e: EmailHeaders, *, uncertain: bool
) -> None:
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
            uncertain,
        ),
    )


def _recount_quoted(
    cur: psycopg.Cursor[TupleRow], documents: Sequence[Document], ids: dict[str, int]
) -> None:
    ## The emails here quote exactly what they quote now; then each quoted message here is
    ## readable by every mailbox holding any email that quotes it, from this day or another.
    emails = [ids[d.external_id] for d in documents if d.kind == "email"]
    quoted = [d for d in documents if d.kind == "quoted"]
    cur.execute("DELETE FROM quotes WHERE email_id = ANY(%s)", (emails,))
    cur.executemany(
        "INSERT INTO quotes (email_id, quoted_id) SELECT unnest(%s::bigint[]), %s",
        [([ids[e] for e in d.quoted_by], ids[d.external_id]) for d in quoted],
    )
    quoted_ids = [ids[d.external_id] for d in quoted]
    cur.execute("DELETE FROM access WHERE document_id = ANY(%s)", (quoted_ids,))
    cur.execute(
        "INSERT INTO access (document_id, mailbox) SELECT DISTINCT q.quoted_id, a.mailbox"
        " FROM quotes q JOIN access a ON a.document_id = q.email_id WHERE q.quoted_id = ANY(%s)",
        (quoted_ids,),
    )


def _rechunk(cur: psycopg.Cursor[TupleRow], document_id: int, texts: list[str]) -> int:
    ## Deleting the chunks takes their vectors with them (ON DELETE CASCADE).
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    cur.execute(
        "INSERT INTO chunks (document_id, position, text, text_hash)"
        " SELECT %s, n - 1, t, h FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY AS u(t, h, n)",
        (document_id, texts, [text_hash(t) for t in texts]),
    )
    return len(texts)
