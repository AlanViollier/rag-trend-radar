"""c-TF-IDF: the words common in one topic and rare in the others, which is what it's about."""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

TOP_WORDS = 10
# Words and years: "1997" names the restatement, other numbers are deal and phone numbers.
_WORD = r"\b(?:[a-z][a-z'-]+|\d{4})\b"


def keywords(topics: Sequence[str], top: int = TOP_WORDS) -> list[list[str]]:
    """Each topic's `top` most distinctive words, best first; `topics` holds one long text each.

    A word scores its share of this topic's words, times how rare it is across all topics: its
    log(1 + average words per topic / its count everywhere).
    """
    if len(topics) < 2:
        return [[] for _ in topics]
    vectorizer = CountVectorizer(stop_words="english", token_pattern=_WORD, lowercase=True)
    counts = vectorizer.fit_transform(topics).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    per_topic = counts.sum(axis=1, keepdims=True)
    rarity = np.log(1 + per_topic.mean() / counts.sum(axis=0))
    scores = counts / np.maximum(per_topic, 1) * rarity
    ranked = np.argsort(-scores, axis=1)[:, :top]
    return [[str(vocabulary[w]) for w in row if counts[i, w] > 0] for i, row in enumerate(ranked)]


def reading_text(chunk: str) -> str:
    """A chunk's subject and body: addresses and dates in its header would top every list."""
    lead, _, body = chunk.partition("\n\n")
    first = lead.partition("\n")[0]
    subject = first.removeprefix("Subject: ") if first.startswith("Subject: ") else ""
    return f"{subject} {body}"
