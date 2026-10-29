"""The store's write side: documents and chunks, then the vectors for them."""

from rag_trend_radar.index.documents import delete_document, upsert_documents
from rag_trend_radar.index.vectors import (
    coverage,
    derive_vectors,
    drop_vectors,
    ensure_vectors,
    from_cache,
    unembedded,
    vectors_identifier,
    vectors_table,
    write_vectors,
)

__all__ = [
    "coverage",
    "delete_document",
    "derive_vectors",
    "drop_vectors",
    "ensure_vectors",
    "from_cache",
    "unembedded",
    "upsert_documents",
    "vectors_identifier",
    "vectors_table",
    "write_vectors",
]
