"""Calm a blob's weekly numbers for the picture. Later mail is allowed here, never in an alert."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def fill(seen: Sequence[int], values: Sequence[float]) -> NDArray[np.float64]:
    """One value per week from the first seen to the last, weeks a blob went missing drawn straight."""
    return np.interp(np.arange(seen[0], seen[-1] + 1), seen, values)


def median(values: NDArray[np.float64], weeks: int) -> NDArray[np.float64]:
    """Each value replaced by the median of the `weeks` either side of it."""
    return np.array(
        [np.median(values[max(0, i - weeks) : i + weeks + 1]) for i in range(len(values))]
    )


def mean(values: NDArray[np.float64], weeks: int) -> NDArray[np.float64]:
    """Each value replaced by the average of the `weeks` either side of it."""
    return np.array(
        [np.mean(values[max(0, i - weeks) : i + weeks + 1]) for i in range(len(values))]
    )
