import pytest

from rag_trend_radar.evals.questions import _one_question


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Who signed the Cargill swap agreement?", True),
        ("Which game did I attend?  Who was Texas playing?", False),
        ("Why did it shift, and what role did the traders play", False),
        ("Where should I send it?\nActually, simplified: where?", False),
        ("Where should I send it? It went missing.</br>", False),
    ],
)
def test_one_question_means_one_line_ending_in_one_question_mark(text: str, ok: bool) -> None:
    assert _one_question(text) is ok
