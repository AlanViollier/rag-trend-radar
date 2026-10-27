"""Text in, vectors out, with any sentence-transformers model named in config."""

from collections.abc import Callable, Sequence
from functools import cache

import numpy as np
from numpy.typing import NDArray


def embed(texts: Sequence[str], model: str, dims: int) -> NDArray[np.float32]:
    """Embed `texts` in batches on the fastest device, truncated to `dims` and normalised."""
    raise NotImplementedError


@cache
def token_counter(model: str) -> Callable[[str], int]:
    """Count tokens the way `model` reads them, so chunk sizes are in its own units.

    Only the tokenizer loads, not the weights: chunking runs before any embedding.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415  (models extra, absent in CI)

    tokenizer = AutoTokenizer.from_pretrained(model)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
