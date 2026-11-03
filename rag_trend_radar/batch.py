"""The daily loop: stage the dataset once, then turn each simulated day into the store's rows.

Re-running a day changes nothing (rule 5): documents upsert by content hash, and the day's
`batches` row is recounted from what's stored, never incremented.
"""

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar import index
from rag_trend_radar.days import cutoff
from rag_trend_radar.ingest import assemble
from rag_trend_radar.ingest.dedup import text_hash
from rag_trend_radar.schema import DayFile, Document, RawFile, Scan
from rag_trend_radar.sources import enron


def stage(conn: psycopg.Connection[TupleRow], root: Path, tz: ZoneInfo) -> Scan:
    """Read every file once: raw files, quarantine, employees, and the replay's schedule.

    Files already in `raw_files` stay as they were (rule 4). `copies` and `boilerplate` are
    derived, so they're rewritten whole: after a parsing change, staging again updates them.
    """
    scanned = assemble.scan(root)
    dates = assemble.resolve(scanned.copies)
    stored = {path for (path,) in conn.execute("SELECT path FROM raw_files").fetchall()}
    paths = {c.path for c in scanned.copies} | {r.path for r in scanned.rejected}
    new = sorted(paths - stored)
    with conn.cursor() as cur:
        with cur.copy("COPY raw_files (path, mailbox, folder, content) FROM STDIN") as copy:
            for path in new:
                raw = enron.read_raw_file(root, root / path)
                copy.write_row((raw.path, raw.mailbox, raw.folder, raw.content))
        cur.executemany(
            "INSERT INTO quarantine (path, reason, detail) VALUES (%s, %s, %s)"
            " ON CONFLICT (path) DO UPDATE SET reason = EXCLUDED.reason, detail = EXCLUDED.detail",
            [(r.path, r.reason, r.detail) for r in scanned.rejected],
        )
        cur.executemany(
            "INSERT INTO employees (mailbox) VALUES (%s) ON CONFLICT DO NOTHING",
            [(m,) for m in scanned.mailboxes],
        )
        cur.execute("TRUNCATE copies, boilerplate")
        with cur.copy(
            "COPY copies (path, email_key, sent_at, date_uncertain, own_hash) FROM STDIN"
        ) as copy:
            for c in scanned.copies:
                copy.write_row((c.path, c.key, *dates[c.key], text_hash(c.own)))
        with cur.copy("COPY boilerplate (line, from_day) FROM STDIN") as copy:
            for line, day in assemble.boilerplate(scanned.copies, dates, tz).items():
                copy.write_row((line, day))
    return scanned


def run_day(
    conn: psycopg.Connection[TupleRow],
    day: date,
    tz: ZoneInfo,
    pieces: Callable[[Document, frozenset[str]], list[str]],
) -> int:
    """Turn `day`'s emails into documents and chunks; return how many chunks need embedding.

    The day's `batches` row opens unfinished here, so a crash leaves it visible and `as_of`
    still defaults to the day before. `finish_day` closes it once the vectors exist.
    """
    conn.execute(
        "INSERT INTO batches (day, started_at) VALUES (%s, now())"
        " ON CONFLICT (day) DO UPDATE SET started_at = now(), finished_at = NULL",
        (day,),
    )
    rows = conn.execute(
        "SELECT r.path, r.mailbox, r.folder, r.content, c.email_key, c.sent_at, c.date_uncertain"
        " FROM copies c JOIN raw_files r USING (path)"
        " WHERE c.sent_at >= %s AND c.sent_at < %s ORDER BY c.sent_at, c.email_key, c.path",
        (cutoff(day - timedelta(days=1)), cutoff(day)),
    ).fetchall()
    files = [
        DayFile(
            raw=RawFile(path=path, mailbox=mailbox, folder=folder, content=bytes(content)),
            key=key,
            ## Chunk headers print the date, so it's read in Houston time, not the session's.
            sent_at=sent_at.astimezone(tz),
            date_uncertain=uncertain,
        )
        for path, mailbox, folder, content, key, sent_at, uncertain in rows
    ]
    frequent = frozenset(
        line
        for (line,) in conn.execute(
            "SELECT line FROM boilerplate WHERE from_day <= %s", (day,)
        ).fetchall()
    )
    documents = assemble.documents(files, lambda hashes: _held(conn, hashes))
    return index.upsert_documents(conn, documents, lambda doc: pieces(doc, frequent))


def finish_day(conn: psycopg.Connection[TupleRow], day: date, model: str, dims: int) -> None:
    """Close `day`'s batches row with its counts recounted from the store."""
    recount_day(conn, day, model, dims)
    conn.execute("UPDATE batches SET finished_at = now() WHERE day = %s", (day,))


def recount_day(conn: psycopg.Connection[TupleRow], day: date, model: str, dims: int) -> None:
    """Recount `day`'s batches row from what's stored: after its run, or after a delete."""
    in_day = "d.deleted_at IS NULL AND d.sent_at >= %(start)s AND d.sent_at < %(end)s"
    conn.execute(
        sql.SQL(
            "UPDATE batches SET"
            " emails_in = (SELECT count(*) FROM documents d WHERE d.kind = 'email' AND {in_day}),"
            " documents_new = (SELECT count(*) FROM documents d WHERE {in_day}),"
            " chunks_embedded = (SELECT count(*) FROM documents d JOIN chunks c"
            "  ON c.document_id = d.id JOIN {vectors} v ON v.chunk_id = c.id WHERE {in_day})"
            " WHERE day = %(day)s"
        ).format(in_day=sql.SQL(in_day), vectors=index.ensure_vectors(conn, model, dims)),
        {"day": day, "start": cutoff(day - timedelta(days=1)), "end": cutoff(day)},
    )


def _held(conn: psycopg.Connection[TupleRow], hashes: set[str]) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT own_hash FROM copies WHERE own_hash = ANY(%s)", (list(hashes),)
    ).fetchall()
    return {digest for (digest,) in rows}
