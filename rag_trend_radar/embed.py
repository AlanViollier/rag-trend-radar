"""Text in, vectors out, with any sentence-transformers model named in config."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def embed(texts: Sequence[str], model: str, dims: int) -> NDArray[np.float32]:
    """Embed `texts` in batches on the fastest device, truncated to `dims` and normalised."""
    raise NotImplementedError
