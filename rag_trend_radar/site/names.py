"""A blob's rough name from its keywords, which changes only when the subject really does."""

from collections import Counter
from collections.abc import Sequence

from rag_trend_radar.trends.blob_kinds import CHATTY

NAME_WORDS = 3
WIRE_NAME = "dow jones clips"
# Weeks either side whose keywords count towards a week's name.
_AROUND = 4
# A blob-week's best words count, the best ones more.
_READ = 6
# A new name must hold for three quarters of the next 8 weeks.
_AHEAD, _HOLD = 8, 0.75
_WIRE = frozenset({"mentions", "dow", "jones", "copyright", "reserved", "rights"})
_SAYS_NOTHING_WORDS = (
    "com www http enron enron's 2000 2001 2002 said corp let pm am ect hou attached please thanks"
    " mentions dow jones copyright reserved rights reuters english image free new comtex"
    " jan feb mar apr may jun jul aug sep oct nov dec"
)
_SAYS_NOTHING = CHATTY | set(_SAYS_NOTHING_WORDS.split())


def rough(seen: Sequence[int], words: Sequence[Sequence[str]]) -> list[tuple[int, str]]:
    """The blob's names as (week it starts, name), given the weeks it was seen and their keywords."""
    first, last = seen[0], seen[-1]
    raw = {week: _words_around(week, seen, words) for week in range(first, last + 1)}
    current = raw[first]
    runs = [(first, _name(current))]
    for week in range(first + 1, last + 1):
        ahead = [raw[w] for w in range(week, min(week + _AHEAD, last + 1))]
        holds = sum(set(later) == set(raw[week]) for later in ahead) >= _HOLD * len(ahead)
        if set(raw[week]) != set(current) and holds:
            current = raw[week]
            runs.append((week, _name(current)))
    return runs


def _name(words: tuple[str, ...]) -> str:
    return " ".join(words)


def _words_around(
    week: int, seen: Sequence[int], words: Sequence[Sequence[str]]
) -> tuple[str, ...]:
    near = [i for i, w in enumerate(seen) if abs(w - week) <= _AROUND] or [
        min(range(len(seen)), key=lambda i: abs(seen[i] - week))
    ]
    wire = [sum(w in _WIRE for w in words[i][:_READ]) >= 2 for i in near]
    score: Counter[str] = Counter()
    for i in near:
        for rank, word in enumerate(words[i][:_READ]):
            if word not in _SAYS_NOTHING:
                score[word] += _READ - rank
    if sum(wire) >= len(wire) / 2 or not score:
        return (WIRE_NAME,)
    return tuple(word for word, _ in score.most_common(NAME_WORDS))
