"""What a blob is doing in a week, from mail up to that week only: size, activity, surge, movement."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rag_trend_radar.trends.frames import Frame
from rag_trend_radar.trends.topics import NOISE, Pool

## A chunk in no blob counts for one when it's closer than the blob's own loosest tenth, and clearly
## nearer than the runner-up. For counting only: linking with them lowers seed agreement.
NEAR_PERCENTILE, NEAR_MARGIN = 10, 0.01
RECENT_DAYS = 28
## Keeps a blob with almost no older mail from dividing by nothing's worth of history.
_PRIOR = 5
## Half-lives in weeks. Centres wobble about as much as they move in one week, so raw weekly steps
## or a blob's speed find mostly noise; two averages apart find a direction it's been holding.
FAST, SLOW = 2, 8


@dataclass(frozen=True)
class Counts:
    """A blob's mail in one week's window."""

    chunks: int
    mail: int
    recent: int
    recent_press: int
    surge: float


def centres(frame: Frame, vectors: NDArray[np.float32]) -> dict[int, NDArray[np.float32]]:
    """Each blob's centre in the full 768 numbers, length 1."""
    window = vectors[frame.rows]
    found = {}
    for label in sorted(set(frame.labels.tolist()) - {NOISE}):
        mean = window[frame.labels == label].mean(axis=0)
        found[label] = mean / np.linalg.norm(mean)
    return found


def counted(frame: Frame, vectors: NDArray[np.float32]) -> NDArray[np.int64]:
    """`frame.labels` with the near chunks given to the blob they're near."""
    labels = frame.labels.copy()
    middle = centres(frame, vectors)
    strays = np.flatnonzero(labels == NOISE)
    if not middle or not len(strays):
        return labels
    numbers = np.array(list(middle))
    window = vectors[frame.rows]
    points = np.array(list(middle.values()))
    edge = np.array(
        [np.percentile(window[labels == n] @ middle[n], NEAR_PERCENTILE) for n in numbers]
    )
    similar = window[strays] @ points.T
    best = similar.argmax(axis=1)
    top = similar[np.arange(len(strays)), best]
    similar[np.arange(len(strays)), best] = -1
    near = (top >= edge[best]) & (top - similar.max(axis=1) >= NEAR_MARGIN)
    labels[strays[near]] = numbers[best[near]]
    return labels


def counts(frame: Frame, chosen: Pool, near: NDArray[np.int64]) -> dict[int, Counts]:
    """Every blob's counts in this week; `near` is `counted`'s labels."""
    days = np.array([d.toordinal() for d in chosen.days])[frame.rows]
    press = np.array(chosen.press, dtype=bool)[frame.rows]
    recent = days > frame.end.toordinal() - RECENT_DAYS
    found = {}
    for label in sorted(set(frame.labels.tolist()) - {NOISE}):
        mail = near == label
        now = int((mail & recent).sum())
        before = int((mail & ~recent).sum())
        found[label] = Counts(
            chunks=int((frame.labels == label).sum()),
            mail=int(mail.sum()),
            recent=now,
            recent_press=int((mail & recent & press).sum()),
            surge=surge(now, before, int(recent.sum()), int((~recent).sum())),
        )
    return found


def surge(recent: int, before: int, all_recent: int, all_before: int) -> float:
    """Chunks above normal: what the blob got, less its usual share of everything that came in.

    Mail swings tenfold in late 2001, so a raw count would call every blob a surge.
    """
    return recent - (before + _PRIOR) * max(all_recent, 1) / max(all_before, 1)


def moving(weeks: Sequence[int], points: Sequence[NDArray[np.float32]]) -> list[float]:
    """How far a blob's recent centre sits from its slower average, per week it was seen."""
    fast = slow = points[0].copy()
    found = []
    for i, (week, point) in enumerate(zip(weeks, points, strict=True)):
        gap = week - weeks[i - 1] if i else 1
        fast = fast * 0.5 ** (gap / FAST) + point * (1 - 0.5 ** (gap / FAST))
        slow = slow * 0.5 ** (gap / SLOW) + point * (1 - 0.5 ** (gap / SLOW))
        found.append(1 - float(fast @ slow / np.linalg.norm(fast) / np.linalg.norm(slow)))
    return found
