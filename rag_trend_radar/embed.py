"""Text in, vectors out, with any sentence-transformers model named in config."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def embed(
    texts: Sequence[str], model: str, dims: int, *, query: bool = False
) -> NDArray[np.float32]:
    """Embed `texts` on the fastest device, truncated to `dims` and normalised.

    Queries and documents are encoded differently when the model was trained that way
    (EmbeddingGemma prefixes `task: search result | query: ` or `title: none | text: `); a model
    with no prompts gets the bare text either way.
    """
    encoder = _load(model)
    encode = encoder.encode_query if query else encoder.encode_document
    vectors = encode(list(texts), truncate_dim=dims, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


@cache
def token_counter(model: str) -> Callable[[str], int]:
    """Count tokens the way `model` reads them, so chunk sizes are in its own units.

    Only the tokenizer loads, not the weights: chunking runs before any embedding.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415  (models extra, absent in CI)

    tokenizer = AutoTokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


@cache
def _load(model: str) -> SentenceTransformer:
    ## float32, the library default: EmbeddingGemma returns NaNs in float16, and Apple M1 has no
    ## native bfloat16.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(model)
