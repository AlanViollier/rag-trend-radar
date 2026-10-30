"""The review page: what the judge decided, laid out for a person to check, as Markdown."""

import random
from collections.abc import Mapping, Sequence

from rag_trend_radar.schema import EvalQuestion, Judgment, RecallRun, Verdict

## The method the app runs; its misses are the ones the README talks about.
APP = "hybrid + rerank"
EXCERPT = 700
SPOT_CHECKS = 20


def review(
    questions: Sequence[EvalQuestion],
    run: RecallRun,
    judgments: Sequence[Judgment],
    texts: Mapping[str, str],
    seed: int = 1,
) -> str:
    """Four sections: other documents judged answering (they move the judged scores), questions
    whose own document failed (dropped), the app's misses at #1, and a seeded sample to spot-check.
    """
    by_pair = {(j.question_id, j.external_id): j for j in judgments}
    asked = {q.id: q for q in questions}
    labels = {(q.id, x) for q in questions for x in q.relevant}
    lines = [f"# Review: {run.manifest.questions}", ""]

    lines += ["## 1. Other documents the judge found answering", ""]
    lines += ["They count in the judged table. Each one: is it really an answer?", ""]
    for j in judgments:
        if (j.question_id, j.external_id) not in labels and j.answers:
            lines += _pair(asked[j.question_id], j, texts, show_label=True)

    lines += ["## 2. Questions whose own document didn't answer", ""]
    lines += ["Dropped from both tables. Each one: bad question, or a judge mistake?", ""]
    for q in questions:
        for x in q.relevant:
            own = by_pair.get((q.id, x))
            if own is not None and not own.answers:
                lines += _pair(q, own, texts, show_label=False)

    lines += [f"## 3. Where {APP} put something else first", ""]
    for q in questions:
        top = run.rankings[APP].get(q.id, [])
        if top and top[0] not in q.relevant:
            first = by_pair.get((q.id, top[0]))
            lines += [f"### {q.id} ({q.person})", "", f"> {q.question}", ""]
            lines += [f"**Labelled** `{x}`" for x in q.relevant]
            lines += [
                "",
                f"**Ranked first** `{top[0]}`: {_verdict(first)}",
                "",
                _excerpt(texts, top[0]),
                "",
            ]

    lines += [f"## 4. Spot-check: {SPOT_CHECKS} judgments drawn at random", ""]
    lines += ["Agree or not, for each. The share you disagree with is the judge's error rate.", ""]
    ## A seeded draw so the page reproduces, not a secret: S311 is about cryptography.
    for j in random.Random(seed).sample(list(judgments), min(SPOT_CHECKS, len(judgments))):  # noqa: S311
        lines += _pair(
            asked[j.question_id], j, texts, show_label=(j.question_id, j.external_id) not in labels
        )
    return "\n".join(lines) + "\n"


def _pair(q: EvalQuestion, j: Judgment, texts: Mapping[str, str], *, show_label: bool) -> list[str]:
    lines = [f"### {q.id} ({q.person})", "", f"> {q.question}", ""]
    if show_label:
        lines += [f"**Labelled** `{x}`" for x in q.relevant] + [""]
    lines += [
        f"**Judged** `{j.external_id}`: {_verdict(j)}",
        "",
        _excerpt(texts, j.external_id),
        "",
    ]
    return lines


def _verdict(j: Judgment | None) -> str:
    if j is None:
        return "not judged"
    found = "" if j.quoted or j.verdict is Verdict.NO else " (quote NOT in the document)"
    quote = f": “{' '.join(j.quote.split())}”" if j.quote else ""
    return f"**{j.verdict}**{found}{quote}"


def _excerpt(texts: Mapping[str, str], external_id: str) -> str:
    text = texts.get(external_id, "(not in the store)")
    cut = text[:EXCERPT] + (" …" if len(text) > EXCERPT else "")
    return "```text\n" + cut + "\n```"
