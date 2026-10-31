import pytest

from rag_trend_radar.evals.copies import is_copy
from rag_trend_radar.evals.retrieval import score

LABEL = (
    "Hotel rooms have been reserved for you at the Doubletree; however each participant is "
    "responsible for payment. Flights should be booked by each participant before Sept 19.",
    "",
)


def test_any_answering_document_first_is_a_hit_where_recall_would_count_half() -> None:
    ranked = {"a": ["9", "1"]}
    assert score({"a": {"1"}}, ranked, ["hit_rate@1"]) == {"hit_rate@1": 0.0}
    assert score({"a": {"1", "9"}}, ranked, ["hit_rate@1"]) == {"hit_rate@1": 1.0}
    assert score({"a": {"1", "9"}}, ranked, ["recall@1"]) == {"recall@1": 0.5}


@pytest.mark.parametrize(
    ("other", "copy"),
    [
        (("fyi..", LABEL[0]), True),
        (
            ("Rooms wait at the Doubletree, you pay them; book flights yourselves by Sept 19.", ""),
            False,
        ),
        ((LABEL[0].replace("Doubletree", "Doubletree Hotel"), ""), True),
        ((LABEL[0].replace("Sept 19", "Sept 26"), ""), False),
        (("Lunch on Friday?", ""), False),
    ],
    ids=[
        "forward-quotes-it",
        "reworded",
        "resent-with-an-edit",
        "template-other-date",
        "unrelated",
    ],
)
def test_a_copy_carries_the_label_or_its_words_and_every_number(
    other: tuple[str, str], copy: bool
) -> None:
    assert is_copy(LABEL, other) is copy


@pytest.mark.parametrize(
    ("other", "copy"),
    [
        (("Deal #798411.1 was entered at $1.", ""), True),
        (("Deal 7984110 is another one.", ""), False),
        (("Re: the annuity", "> Deal 798411 for 9/13"), True),
    ],
    ids=["holds-it", "inside-a-longer-number", "in-a-quoted-reply"],
)
def test_for_a_code_question_the_code_decides(other: tuple[str, str], copy: bool) -> None:
    assert is_copy(("Deal 798411 annuity", ""), other, code="798411") is copy
