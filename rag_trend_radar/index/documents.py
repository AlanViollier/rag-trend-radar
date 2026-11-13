"""Write documents and their chunks, who may read them, and carry deletes through."""

from collections.abc import Callable, Sequence
from datetime import datetime

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.ingest.chunk import reply_quote
from rag_trend_radar.ingest.dedup import text_hash
from rag_trend_radar.records import Document, EmailHeaders


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
    stored email quoting it, whichever day quoted it, so a re-run changes nothing, and its chunks are cut on the day
    of its first quote, as its title is kept from it. A deleted document is skipped:
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
            document_id, first_sent = _upsert_document(cur, doc)
            ids[doc.external_id] = document_id
            if doc.email is not None:
                _upsert_email(cur, document_id, doc.email, uncertain=doc.date_uncertain)
            if doc.kind != "quoted":
                cur.execute("DELETE FROM access WHERE document_id = %s", (document_id,))
                cur.execute(
                    "INSERT INTO access (document_id, mailbox, from_at)"
                    " SELECT %s, unnest(%s::text[]), %s",
                    (document_id, list(doc.readers), doc.sent_at),
                )
            if doc.paths:
                cur.execute(
                    "UPDATE raw_files SET document_id = %s WHERE path = ANY(%s)",
                    (document_id, list(doc.paths)),
                )
            # Only the first quote cuts a quoted message's chunks: a later one's subject or
            # spelling of the sender would flip them on every replay.
            if first_sent < doc.sent_at:
                continue
            texts = pieces(doc)
            if [text_hash(t) for t in texts] != chunked.get(doc.external_id, []):
                written += _rechunk(cur, document_id, texts, reply_quote(doc))
        _record_quotes(cur, live, ids)
    return written


def delete_document(conn: psycopg.Connection[TupleRow], document_id: int) -> list[datetime]:
    """Take a document out of every search, for every day, and return the dates to recount.

    Its chunks and vectors go, its text and headers are blanked, and the row stays as a
    tombstone so a replay skips it. A quoted message that only this email quoted goes with it;
    one that others quote too keeps the readers who still hold a quote. `raw_files` keeps the
    file as read; it's never edited. Returns the sent_at of every document deleted.
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


def _upsert_document(cur: psycopg.Cursor[TupleRow], doc: Document) -> tuple[int, datetime]:
    # A quoted message comes back each day someone quotes it: it keeps the date and subject of
    # its first quote. An email's date is resolved once, so for it LEAST changes nothing.
    cur.execute(
        "INSERT INTO documents (source, external_id, kind, sent_at, title, body, quoted)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (source, external_id) DO UPDATE SET kind = EXCLUDED.kind,"
        " sent_at = LEAST(documents.sent_at, EXCLUDED.sent_at), title = CASE"
        " WHEN EXCLUDED.sent_at < documents.sent_at THEN EXCLUDED.title ELSE documents.title END,"
        " body = EXCLUDED.body,"
        " quoted = EXCLUDED.quoted"
        " RETURNING id, sent_at",
        (
            doc.source,
            doc.external_id,
            doc.kind,
            doc.sent_at,
            doc.title,
            doc.body,
            doc.quoted,
        ),
    )
    document_id, sent_at = cur.fetchall()[0]
    return int(document_id), sent_at


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
    # The emails here quote exactly what they quote now (a deleted one quotes nothing).
    emails = [ids[d.external_id] for d in documents if d.kind == "email"]
    quoted = [d for d in documents if d.kind == "quoted"]
    cur.execute("DELETE FROM quotes WHERE email_id = ANY(%s)", (emails,))
    cur.executemany(
        "INSERT INTO quotes (email_id, quoted_id) SELECT unnest(%s::bigint[]), %s",
        [([ids[e] for e in d.quoted_by if e in ids], ids[d.external_id]) for d in quoted],
    )
    _recount_access(cur, [ids[d.external_id] for d in quoted])
    # A quoted message no email quotes any more (a cleaner fix changed its text, so it came
    # back under a new id) goes: a replay after a code change leaves no stale copy behind.
    cur.execute(
        "DELETE FROM documents WHERE kind = 'quoted' AND deleted_at IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM quotes q WHERE q.quoted_id = documents.id)"
    )


def _recount_access(cur: psycopg.Cursor[TupleRow], quoted_ids: list[int]) -> None:
    # A quoted message is readable by every mailbox holding an email that quotes it, from the
    # first of those emails on: recounted from quotes, never added to.
    cur.execute("DELETE FROM access WHERE document_id = ANY(%s)", (quoted_ids,))
    cur.execute(
        "INSERT INTO access (document_id, mailbox, from_at)"
        " SELECT q.quoted_id, a.mailbox, min(a.from_at)"
        " FROM quotes q JOIN access a ON a.document_id = q.email_id WHERE q.quoted_id = ANY(%s)"
        " GROUP BY q.quoted_id, a.mailbox",
        (quoted_ids,),
    )


def _rechunk(cur: psycopg.Cursor[TupleRow], document_id: int, texts: list[str], quote: str) -> int:
    # Deleting the chunks takes their vectors with them (ON DELETE CASCADE).
    cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    cur.execute(
        "INSERT INTO chunks (document_id, position, text, text_hash, quote_chars)"
        " SELECT %s, n - 1, t, h, q"
        " FROM unnest(%s::text[], %s::text[], %s::int[]) WITH ORDINALITY AS u(t, h, q, n)",
        (
            document_id,
            texts,
            [text_hash(t) for t in texts],
            [len(quote) + 2 if quote and t.endswith(f"\n\n{quote}") else 0 for t in texts],
        ),
    )
    return len(texts)
