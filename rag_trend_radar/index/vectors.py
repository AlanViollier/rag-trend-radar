"""One vectors table per model and size, filled from the embedding cache or the model."""

import re
from collections.abc import Sequence

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar.config import get_settings


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
    # Readers see a vector only when they see its chunk; the grant is by name like every table.
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
