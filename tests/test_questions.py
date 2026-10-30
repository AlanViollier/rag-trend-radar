from datetime import date

import pytest

from rag_trend_radar.evals.questions import WORDS_RULES, Retype, Style, _one_question, retyped
from rag_trend_radar.schema import EvalQuestion


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Who signed the Cargill swap agreement?", True),
        ("Which game did I attend?  Who was Texas playing?", False),
        ("Why did it shift, and what role did the traders play", False),
        ("Where should I send it?\nActually, simplified: where?", False),
        ("Where should I send it? It went missing.</br>", False),
        ("What correction is needed on Deal #697589.1, and what was the day total?", False),
        ("Who approved it or when was it sent?", False),
        ("What did the West desk and the East desk agree on?", True),
    ],
)
def test_one_question_means_one_line_ending_in_one_question_mark(text: str, ok: bool) -> None:
    assert _one_question(text) is ok


def test_every_style_has_its_wording_rule() -> None:
    assert set(WORDS_RULES) == set(Style)


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        (Retype.DASH, "What is deal NG-617-161 for?"),
        (Retype.SPACE, "What is deal NG-617 161 for?"),
        (Retype.HASH, "What is deal #NG-617161 for?"),
    ],
)
def test_retyping_splits_the_digit_run_or_leads_with_a_hash(how: Retype, expected: str) -> None:
    asked = EvalQuestion(
        id="q",
        question="What is deal NG-617161 for?",
        as_of=date(2001, 9, 30),
        person="lay-k",
        relevant=("x",),
        code="NG-617161",
    )
    assert retyped(asked, how).question == expected
