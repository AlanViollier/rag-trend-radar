"""Text in, vectors out, with any sentence-transformers model named in config."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.config import get_settings
from rag_trend_radar.exceptions import EmbeddingFailed

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


## Apple's GPU, out of memory, returns a whole batch of normalised garbage without raising
## (2026-10-29: batch 128 on long chunks, every row off, norms 1.000, no NaN). Only a known
## answer shows it, so every GPU batch carries this text and its vector must come back as when
## it was embedded alone. 0.99: batch 8 against 32 gave no row below it on the longest chunks.
SENTINEL = "Subject: Sentinel\n\nThe quarterly gas storage report is attached for review."
## Texts per GPU batch, the sentinel included. 64 and up ran out of memory on 512-token chunks.
GPU_BATCH = 32
_MATCH = 0.99


def embed(
    texts: Sequence[str],
    model: str,
    dims: int,
    *,
    query: bool = False,
    skip_failed: bool = False,
) -> NDArray[np.float32]:
    """Embed `texts` on the fastest device, truncated to `dims` and normalised.

    Queries and documents are encoded differently when the model was trained that way
    (EmbeddingGemma prefixes `task: search result | query: ` or `title: none | text: `); a model
    with no prompts gets the bare text either way.

    A GPU batch whose sentinel comes back wrong is split in half and tried again, down to one
    text. A text that fails alone, with the sentinel still right alone, comes back as a NaN row
    with `skip_failed` (a long run goes on; the next run tries it again) and raises
    `EmbeddingFailed` without. When the sentinel fails alone too, the model is broken for every
    text: that always raises.
    """
    ## Sorted by length, so a GPU batch pads to texts of its own size: unsorted, the guard's
    ## small batches ran at half speed (6.3 texts/s against 12.7).
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    ranked = [texts[i] for i in order]
    groups = [ranked[i : i + GPU_BATCH - 1] for i in range(0, len(ranked), GPU_BATCH - 1)]
    vectors = np.zeros((len(texts), dims), dtype=np.float32)
    if groups:
        vectors[order] = np.concatenate([_verified(g, model, dims, query) for g in groups])
        ## Apple's GPU keeps each batch's memory otherwise: 6.8 GB against 1.9 over 15,000 long
        ## texts, at the same speed (2026-11-03), and past 90% of a full embed, swapping.
        _free_gpu()
    if not skip_failed and np.isnan(vectors).any():
        raise EmbeddingFailed(f"{model}: a text failed the sentinel check even alone")
    return vectors


@cache
def token_counter(model: str) -> Callable[[str], int]:
    """Count tokens the way `model` reads them, so chunk sizes are in its own units.

    Only the tokenizer loads, not the weights: chunking runs before any embedding.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415  (models extra, absent in CI)

    tokenizer = AutoTokenizer.from_pretrained(model, revision=get_settings().revision(model))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _verified(texts: list[str], model: str, dims: int, query: bool) -> NDArray[np.float32]:
    ## Out of memory is a batch too large, so smaller batches are the cure, not a pause.
    out = _encode([*texts, SENTINEL], model, dims, query)
    if np.isfinite(out).all() and float(out[-1] @ _sentinel(model, dims, query)) >= _MATCH:
        return out[:-1]
    _free_gpu()
    if len(texts) > 1:
        half = len(texts) // 2
        return np.concatenate(
            [
                _verified(texts[:half], model, dims, query),
                _verified(texts[half:], model, dims, query),
            ]
        )
    alone = _encode([SENTINEL], model, dims, query)[0]
    if float(alone @ _sentinel(model, dims, query)) < _MATCH:
        raise EmbeddingFailed(f"{model}: the sentinel fails alone, every vector would be wrong")
    return np.full((1, dims), np.nan, dtype=np.float32)


def _free_gpu() -> None:
    import torch  # noqa: PLC0415  (models extra, absent in CI)

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@cache
def _sentinel(model: str, dims: int, query: bool) -> NDArray[np.float32]:
    ## Alone, a batch of one: far below any memory limit.
    return np.asarray(_encode([SENTINEL], model, dims, query)[0], dtype=np.float32)


def _encode(texts: list[str], model: str, dims: int, query: bool) -> NDArray[np.float32]:
    encoder = _load(model)
    encode = encoder.encode_query if query else encoder.encode_document
    vectors = encode(texts, truncate_dim=dims, normalize_embeddings=True, batch_size=GPU_BATCH)
    return np.asarray(vectors, dtype=np.float32)


@cache
def _load(model: str) -> SentenceTransformer:
    ## float32, the library default: EmbeddingGemma returns NaNs in float16, and Apple M1 has no
    ## native bfloat16.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(model, revision=get_settings().revision(model))
