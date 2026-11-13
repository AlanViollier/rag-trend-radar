"""When a cluster earns an alert: it grows fast against its own normal, judged on mail up to its week.

Tuned by eye over the whole run (owner 2026-11-07): about one alert every 9 days. Whether one
mattered is read in the naming pass, as an investigator would.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from rag_trend_radar.radar import cluster_kinds

# Times its own normal: a cluster that doubles is busy, one that quadruples is news.
GROWTH = 4
# Share of all the radar's mail in the same 4 weeks, so the bar follows how busy the company is.
FLOOR = 0.01
QUIET_WEEKS = 8


@dataclass(frozen=True)
class Week:
    """One cluster in one week, as an alert sees it then."""

    index: int
    week: date
    recent: int
    surge: float
    other_seed: bool
    trait: cluster_kinds.Trait


@dataclass(frozen=True)
class Alert:
    """A cluster growing fast on `day`: `observed` chunks in the last 4 weeks against `expected` normal."""

    day: date
    cluster: int
    observed: int
    expected: float


def surges(lives: Mapping[int, Sequence[Week]], everything: Mapping[date, int]) -> list[Alert]:
    """Every growth alert, oldest first; `everything` is all radar mail in each week's last 4 weeks.

    A cluster's kind is judged on its weeks so far, never its whole life.
    """
    raised = []
    for cluster, weeks in lives.items():
        fired = -QUIET_WEEKS - 1
        for seen, w in enumerate(weeks):
            normal = w.recent - w.surge
            if (
                w.recent >= GROWTH * normal
                and w.recent >= FLOOR * everything[w.week]
                and w.other_seed
                and w.index - fired > QUIET_WEEKS
                and cluster_kinds.kind([x.trait for x in weeks[: seen + 1]])
                is cluster_kinds.Kind.TOPIC
            ):
                raised.append(Alert(w.week, cluster, w.recent, normal))
                fired = w.index
    return sorted(raised, key=lambda a: (a.day, a.cluster))
