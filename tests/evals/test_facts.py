from datetime import date

from rag_trend_radar.evals.facts import check
from rag_trend_radar.records import FactQuestion
from tests.factories import hit

MAIL = "Subject: Raptors\nFrom: vince.kaminski@enron.com\nDate: 2001-12-20\n\nIt disappeared from my mailbox."
PRESS = "Subject: Enron Mentions\n\nThe memo said it could implode in a wave of scandals."


def _question(**fields: object) -> FactQuestion:
    return FactQuestion.model_validate(
        {"id": "q", "question": "?", "as_of": date(2002, 12, 31), **fields}
    )


def test_a_fact_found_anywhere_in_the_results_passes_regardless_of_case() -> None:
    assert (
        check(_question(facts=["DISAPPEARED"]), [hit(1, PRESS, 0.5), hit(1, MAIL, 0.5)]).missing
        == ()
    )


def test_a_missing_fact_is_named() -> None:
    assert check(_question(facts=["shred"]), [hit(1, MAIL, 0.5)]).missing == ("shred",)


def test_the_chunks_lead_counts_so_a_date_can_be_a_fact() -> None:
    assert check(_question(facts=["2001-12-20"]), [hit(1, MAIL, 0.5)]).missing == ()


def test_a_press_report_doesnt_count_when_the_question_needs_mail() -> None:
    needs = _question(facts=["implode"], needs_mail=True)
    assert check(needs, [hit(1, PRESS, 0.5)]).missing == ("implode",)
    assert check(_question(facts=["implode"]), [hit(1, PRESS, 0.5)]).missing == ()


def test_forbidden_text_counts_from_any_source() -> None:
    assert check(_question(forbid=["implode"]), [hit(1, PRESS, 0.5)]).leaked == ("implode",)


def test_no_results_score_zero_and_miss_everything() -> None:
    result = check(_question(facts=["x"]), [])
    assert (result.missing, result.top_score, result.mail) == (("x",), 0.0, 0)


def test_a_fact_in_a_short_replys_quote_counts_since_the_answer_reads_it() -> None:
    reply = hit(1, MAIL, 0.5).model_copy(update={"context": "Tickets to Judge Porteous"})
    assert check(_question(facts=["Porteous"], needs_mail=True), [reply]).missing == ()
