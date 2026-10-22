"""Write documents, chunks and vectors; carry deletes through to everything derived."""

from collections.abc import Sequence

import numpy as np
import psycopg
from numpy.typing import NDArray
from psycopg.rows import TupleRow

from rag_trend_radar.schema import Document


def upsert_documents(conn: psycopg.Connection[TupleRow], documents: Sequence[Document]) -> int:
    """Insert new documents and re-chunk changed ones; return how many chunks need embedding."""
    raise NotImplementedError


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
