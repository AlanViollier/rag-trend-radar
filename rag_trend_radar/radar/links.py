"""Which blob this week is which blob last week: by the chunks they share, chained into lineages."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from rag_trend_radar.radar.clustering import NOISE
from rag_trend_radar.radar.frames import Frame

# Same blob: each other's biggest overlap, with this share of each side. Weaker counts as nothing:
# seed noise alone produces links under 10 chunks.
SHARE, MIN_SHARED = 0.3, 10
# Eight weeks of absence at most: a blob near the size limit drops out and returns.
BRIDGE = 7


@dataclass(frozen=True)
class Overlap:
    """Chunks two blobs share, and that as a share of what each side shared with anything."""

    shared: int
    of_first: float
    of_second: float


def overlaps(first: Frame, second: Frame) -> dict[tuple[int, int], Overlap]:
    """Every pair of blobs sharing enough chunks, from the rows in a blob in both frames."""
    _, in_first, in_second = np.intersect1d(first.rows, second.rows, return_indices=True)
    a, b = first.labels[in_first], second.labels[in_second]
    placed = (a != NOISE) & (b != NOISE)
    if not placed.any():
        return {}
    pairs, counts = np.unique(np.stack([a[placed], b[placed]], axis=1), axis=0, return_counts=True)
    out: defaultdict[int, int] = defaultdict(int)
    into: defaultdict[int, int] = defaultdict(int)
    for (u, v), n in zip(pairs.tolist(), counts.tolist(), strict=True):
        out[u] += n
        into[v] += n
    return {
        (u, v): Overlap(n, n / out[u], n / into[v])
        for (u, v), n in zip(pairs.tolist(), counts.tolist(), strict=True)
        if n >= MIN_SHARED
    }


def same(first: Frame, second: Frame) -> dict[int, int]:
    """Second-frame blob → the first-frame blob it continues."""
    found = overlaps(first, second)
    best_out: dict[int, int] = {}
    best_in: dict[int, int] = {}
    for (u, v), overlap in found.items():
        if u not in best_out or overlap.shared > found[(u, best_out[u])].shared:
            best_out[u] = v
        if v not in best_in or overlap.shared > found[(best_in[v], v)].shared:
            best_in[v] = u
    return {
        v: u
        for (u, v), overlap in found.items()
        if best_out[u] == v
        and best_in[v] == u
        and overlap.of_first >= SHARE
        and overlap.of_second >= SHARE
    }


def lineages(frames: Sequence[Frame], bridge: int = BRIDGE) -> dict[tuple[date, int], int]:
    """A lineage number per (week, blob), going forward only: no later week changes an earlier one."""
    found: dict[tuple[date, int], int] = {}
    fresh = 0
    for i, frame in enumerate(frames):
        continued = same(frames[i - 1], frame) if i else {}
        for label in _blobs(frame):
            if label in continued:
                found[(frame.end, label)] = found[(frames[i - 1].end, continued[label])]
            else:
                found[(frame.end, label)] = fresh
                fresh += 1
        if not i:
            continue
        settled = set(continued)
        taken = {found[(frames[i - 1].end, label)] for label in _blobs(frames[i - 1])}
        for back in range(2, min(bridge, i - 1) + 2):
            earlier = frames[i - back]
            for v, u in same(earlier, frame).items():
                lineage = found[(earlier.end, u)]
                if v not in settled and lineage not in taken:
                    found[(frame.end, v)] = lineage
                    settled.add(v)
                    taken.add(lineage)
            taken |= {found[(earlier.end, label)] for label in _blobs(earlier)}
    return found


def _blobs(frame: Frame) -> list[int]:
    return sorted(set(frame.labels.tolist()) - {NOISE})
