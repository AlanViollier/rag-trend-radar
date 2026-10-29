"""Write documents, chunks and vectors; carry deletes through to everything derived."""

import re
from collections.abc import Callable, Sequence
from datetime import datetime

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
    """Insert new documents and re-chunk any whose chunk texts changed; return chunks written.

    Chunks are compared as texts, not by the body: a change to the quoted part alone keeps them
    (and their vectors), a new boilerplate line or chunk size rewrites them. Who may read an
    email is rewritten every time from `readers`, so a copy found in one more mailbox widens
    access without touching the chunks. A quoted message's readers are recounted from every
    stored email quoting it, whichever day quoted it (rule 5). A deleted document is skipped:
    replaying its day doesn't bring it back.
    """
    external_ids = [d.external_id for d in documents]
    deleted = {
        external_id
        for (external_id,) in conn.execute(
            "SELECT external_id FROM documents WHERE external_id = ANY(%s)"
            " AND deleted_at IS NOT NULL",
            (external_ids,),
        ).fetchall()
    }
    chunked: dict[str, list[str]] = dict(
        conn.execute(
            "SELECT d.external_id, array_agg(c.text_hash ORDER BY c.position) FROM documents d"
            " JOIN chunks c ON c.document_id = d.id WHERE d.external_id = ANY(%s)"
            " GROUP BY d.external_id",
            (external_ids,),
        ).fetchall()
    )
    live = [d for d in documents if d.external_id not in deleted]
    ids: dict[str, int] = {}
    written = 0
    with conn.cursor() as cur:
        for doc in live:
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
            texts = pieces(doc)
            if [text_hash(t) for t in texts] != chunked.get(doc.external_id, []):
                written += _rechunk(cur, document_id, texts)
        _record_quotes(cur, live, ids)
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


def coverage(conn: psycopg.Connection[TupleRow]) -> dict[str, int]:
    """Every vectors table in the configured schema and how many chunks it holds a vector for.

    A model swap flips config only once the new table covers every chunk: until then the old
    table keeps serving searches and the new one fills beside it.
    """
    tables = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = %s AND tablename LIKE 'vectors\\_%%'"
        " ORDER BY tablename",
        (get_settings().db_schema,),
    ).fetchall()
    return {
        name: conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(
                sql.Identifier(get_settings().db_schema, name)
            )
        ).fetchall()[0][0]
        for (name,) in tables
    }


def drop_vectors(conn: psycopg.Connection[TupleRow], model: str, dims: int) -> None:
    """Drop `model`'s vectors at `dims`, the last step of a swap away from it."""
    conn.execute(sql.SQL("DROP TABLE {}").format(vectors_identifier(model, dims)))


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


def from_cache(conn: psycopg.Connection[TupleRow], model: str, dims: int) -> int:
    """Give every chunk missing a vector the cached one for its exact text; return how many.

    The cache outlives resets and is shared by experiments, so a rebuild or a re-chunked
    variant embeds only text no run has embedded with this model, revision and size.
    """
    table = ensure_vectors(conn, model, dims)
    cur = conn.execute(
        sql.SQL(
            "INSERT INTO {t} (chunk_id, embedding)"
            " SELECT c.id, e.embedding FROM chunks c JOIN cache.embeddings e"
            " ON e.text_hash = c.text_hash AND e.model = %s AND e.revision = %s AND e.dims = %s"
            " WHERE NOT EXISTS (SELECT 1 FROM {t} v WHERE v.chunk_id = c.id)"
            " ON CONFLICT (chunk_id) DO NOTHING"
        ).format(t=table),
        (model, get_settings().revision(model), dims),
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
    """Store vectors in vectors_<model>_<dims> (created on first use) and in the cache."""
    table = ensure_vectors(conn, model, vectors.shape[1])
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "INSERT INTO {} (chunk_id, embedding) VALUES (%s, %s)"
                " ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ).format(table),
            [(i, HalfVector(v)) for i, v in zip(chunk_ids, vectors, strict=True)],
        )
        cur.execute(
            sql.SQL(
                "INSERT INTO cache.embeddings (model, revision, dims, text_hash, embedding)"
                " SELECT DISTINCT ON (c.text_hash) %s, %s, %s, c.text_hash, v.embedding"
                " FROM chunks c JOIN {} v ON v.chunk_id = c.id WHERE c.id = ANY(%s)"
                " ORDER BY c.text_hash ON CONFLICT DO NOTHING"
            ).format(table),
            (model, get_settings().revision(model), vectors.shape[1], list(chunk_ids)),
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


def delete_document(conn: psycopg.Connection[TupleRow], document_id: int) -> list[datetime]:
    """Take a document out of every search, for every day, and return the dates to recount.

    Its chunks and vectors go, its text and headers are blanked, and the row stays as a
    tombstone so a replay skips it. A quoted message that only this email quoted goes with it;
    one that others quote too keeps the readers who still hold a quote. `raw_files` keeps the
    file as read (rule 4). Returns the sent_at of every document deleted.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM quotes WHERE email_id = %s RETURNING quoted_id", (document_id,))
        quoted = [i for (i,) in cur.fetchall()]
        cur.execute(
            "SELECT id FROM documents WHERE id = ANY(%s)"
            " AND NOT EXISTS (SELECT 1 FROM quotes q WHERE q.quoted_id = documents.id)",
            (quoted,),
        )
        gone = [document_id, *(i for (i,) in cur.fetchall())]
        for table, column in (
            ("chunks", "document_id"),
            ("emails", "document_id"),
            ("access", "document_id"),
            ("quotes", "quoted_id"),
        ):
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE {} = ANY(%s)").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (gone,),
            )
        cur.execute(
            "UPDATE documents SET deleted_at = now(), title = '', body = '', quoted = ''"
            " WHERE id = ANY(%s) RETURNING sent_at",
            (gone,),
        )
        sent = [sent_at for (sent_at,) in cur.fetchall()]
        _recount_access(cur, [i for i in quoted if i not in gone])
    return sent


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


def _record_quotes(
    cur: psycopg.Cursor[TupleRow], documents: Sequence[Document], ids: dict[str, int]
) -> None:
    ## The emails here quote exactly what they quote now (a deleted one quotes nothing).
    emails = [ids[d.external_id] for d in documents if d.kind == "email"]
    quoted = [d for d in documents if d.kind == "quoted"]
    cur.execute("DELETE FROM quotes WHERE email_id = ANY(%s)", (emails,))
    cur.executemany(
        "INSERT INTO quotes (email_id, quoted_id) SELECT unnest(%s::bigint[]), %s",
        [([ids[e] for e in d.quoted_by if e in ids], ids[d.external_id]) for d in quoted],
    )
    _recount_access(cur, [ids[d.external_id] for d in quoted])
    ## A quoted message no email quotes any more (a cleaner fix changed its text, so it came
    ## back under a new id) goes: a replay after a code change leaves no stale copy behind.
    cur.execute(
        "DELETE FROM documents WHERE kind = 'quoted' AND deleted_at IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM quotes q WHERE q.quoted_id = documents.id)"
    )


def _recount_access(cur: psycopg.Cursor[TupleRow], quoted_ids: list[int]) -> None:
    ## A quoted message is readable by every mailbox holding an email that quotes it, from any
    ## day: recounted from quotes, never added to.
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
