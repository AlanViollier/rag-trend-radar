from datetime import date

from rag_trend_radar.evals.facts import check
from rag_trend_radar.schema import FactQuestion, SearchHit

MAIL = "Subject: Raptors\nFrom: vince.kaminski@enron.com\nDate: 2001-12-20\n\nIt disappeared from my mailbox."
PRESS = "Subject: Enron Mentions\n\nThe memo said it could implode in a wave of scandals."


def _hit(text: str, score: float = 0.5) -> SearchHit:
    return SearchHit(chunk_id=1, document_id=1, text=text, score=score)


def _question(**fields: object) -> FactQuestion:
    return FactQuestion.model_validate(
        {"id": "q", "question": "?", "as_of": date(2002, 12, 31), **fields}
    )


def test_a_fact_found_anywhere_in_the_results_passes_regardless_of_case() -> None:
    assert check(_question(facts=["DISAPPEARED"]), [_hit(PRESS), _hit(MAIL)]).missing == ()


def test_a_missing_fact_is_named() -> None:
    assert check(_question(facts=["shred"]), [_hit(MAIL)]).missing == ("shred",)


def test_the_chunks_lead_counts_so_a_date_can_be_a_fact() -> None:
    assert check(_question(facts=["2001-12-20"]), [_hit(MAIL)]).missing == ()


def test_a_press_report_doesnt_count_when_the_question_needs_mail() -> None:
    needs = _question(facts=["implode"], needs_mail=True)
    assert check(needs, [_hit(PRESS)]).missing == ("implode",)
    assert check(_question(facts=["implode"]), [_hit(PRESS)]).missing == ()


def test_forbidden_text_counts_from_any_source() -> None:
    assert check(_question(forbid=["implode"]), [_hit(PRESS)]).leaked == ("implode",)


def test_no_results_score_zero_and_miss_everything() -> None:
    result = check(_question(facts=["x"]), [])
    assert (result.missing, result.top_score, result.mail) == (("x",), 0.0, 0)
