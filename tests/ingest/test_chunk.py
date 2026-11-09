from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from rag_trend_radar.ingest.chunk import (
    QUOTE_CHARS,
    SHORT_REPLY,
    chunk,
    header,
    pieces,
    reply_quote,
)
from rag_trend_radar.schema import Document


def _words(text: str) -> int:
    return len(text.split())


def test_a_short_email_is_one_piece_led_by_its_header() -> None:
    lead = header("lunch", "pat.rowe@enron.com", datetime(2001, 5, 9, 15, 12))
    assert chunk("Sounds good, see you Thursday.", _words, lead=lead) == [
        "Subject: lunch\nFrom: pat.rowe@enron.com\nDate: 2001-05-09\n\n"
        "Sounds good, see you Thursday."
    ]


def test_unknown_header_fields_are_left_out() -> None:
    assert header("lunch") == "Subject: lunch"
    assert header("lunch", "Pat Rowe") == "Subject: lunch\nFrom: Pat Rowe"


def test_a_quoted_message_leads_without_the_date_it_was_first_quoted() -> None:
    doc = Document(
        source="enron",
        external_id="h",
        kind="quoted",
        sent_at=datetime(2001, 9, 3),
        title="Gas",
        author="Pat Rowe",
        body="Prices rose.\n\nPat Rowe\nEnron",
        quoted="",
        public=False,
        readers=("kim-d",),
    )
    assert pieces(doc, _words, frozenset({"Pat Rowe", "Enron"})) == [
        "Subject: Gas\nFrom: Pat Rowe\n\nPrices rose."
    ]


def test_no_lead_means_the_bare_text() -> None:
    assert chunk("Sounds good.", _words) == ["Sounds good."]


def test_empty_own_text_has_no_pieces() -> None:
    assert chunk(" \n ", _words, lead="Subject: fwd") == []


def test_a_long_email_splits_at_paragraphs_first() -> None:
    gas, party = "Gas prices rose again in California. " * 3, "The party is on Friday. " * 3
    pieces = chunk(f"{gas.strip()}\n\n{party.strip()}", _words, max_tokens=20)
    assert [piece.split(".")[0] for piece in pieces] == [
        "Gas prices rose again in California",
        "The party is on Friday",
    ]


# Property: pieces fit, carry the lead, and hold every word of the text once, in order.

_text = st.lists(st.text(alphabet="abcdefg", min_size=1, max_size=8), min_size=1, max_size=300).map(
    lambda words: " ".join(words)
)


@settings(max_examples=100)
@given(text=_text, max_tokens=st.integers(min_value=12, max_value=80))
def test_pieces_fit_and_lose_nothing(text: str, max_tokens: int) -> None:
    lead = "Subject: s\nFrom: a\nDate: 2001-05-09"
    pieces = chunk(text, _words, lead=lead, max_tokens=max_tokens)
    assert all(piece.startswith(lead + "\n\n") and _words(piece) <= max_tokens for piece in pieces)
    assert [w for piece in pieces for w in piece.removeprefix(lead).split()] == text.split()


def _reply(body: str, quoted: str, kind: str = "email") -> Document:
    return Document(
        source="test",
        external_id="r",
        kind=kind,
        sent_at=datetime(2001, 9, 3),
        title="Re: Budget",
        author="",
        body=body,
        quoted=quoted,
        public=False,
        readers=(),
    )


ABOVE = "-----Original Message-----\nFrom: Rowe, Pat\nSubject: Budget\n\nDo we approve the budget?"


def test_a_short_reply_carries_what_it_answered_without_the_quote_header() -> None:
    assert reply_quote(_reply("Yes, approved.", ABOVE)) == "Do we approve the budget?"
    assert pieces(_reply("Yes, approved.", ABOVE), len, frozenset())[0].endswith(
        "Yes, approved.\n\nDo we approve the budget?"
    )


def test_longer_mail_a_quoted_document_and_mail_quoting_nothing_stay_as_they_are() -> None:
    assert reply_quote(_reply("x" * SHORT_REPLY, ABOVE)) == ""
    assert reply_quote(_reply("Yes.", ABOVE, kind="quoted")) == ""
    assert reply_quote(_reply("Yes.", "")) == ""


def test_a_long_quote_is_cut() -> None:
    assert (
        len(
            reply_quote(
                _reply("Yes.", f"-----Original Message-----\n\n{'y' * (QUOTE_CHARS + 500)}")
            )
        )
        == QUOTE_CHARS
    )


def test_a_quote_that_wont_fit_the_budget_is_left_out() -> None:
    assert pieces(_reply("Yes.", ABOVE), len, frozenset(), max_tokens=60) == [
        "Subject: Re: Budget\nDate: 2001-09-03\n\nYes."
    ]


def test_a_reply_with_no_text_of_its_own_has_no_pieces() -> None:
    assert pieces(_reply("", ABOVE), len, frozenset()) == []
