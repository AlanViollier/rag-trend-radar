from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_trend_radar.ingest.clean import (
    block_author,
    block_subject,
    frequent_lines,
    normalize,
    split_quoted,
    strip_boilerplate,
)

## Hand-written in the shapes the Enron export uses; no text copied from the corpus.

OUTLOOK_REPLY = """Sounds good, see you Thursday.

Dana

 -----Original Message-----
From: \tPat Rowe
Sent:\tWednesday, May 09, 2001 3:12 PM
To:\tDana Kim
Subject:\tlunch

Are we still on for lunch?"""

LOTUS_FORWARD = """FYI, see the note below.

---------------------- Forwarded by Dana Kim/HOU/ECT on 10/16/2000
01:42 PM ---------------------------


Pat Rowe@ENRON
10/12/2000 01:12 PM
To: Dana Kim/HOU/ECT@ECT, Lee Park/HOU/ECT@ECT, Sam Ortiz/HOU/ECT@ECT, Jo
Adams/HOU/ECT@ECT
cc:
Subject: storage meeting

The storage meeting moves to room 3270."""

LOTUS_INLINE_FROM = """Agreed.



"Rowe, Pat" <pat.rowe@example.com> on 10/12/2000 01:12:21 PM
To: Dana Kim/HOU/ECT@ECT
cc:
Subject: terms

Please confirm the collar terms."""

NESTED = """Top reply.

-----Original Message-----
From: Lee Park
Sent: Monday, May 07, 2001 9:00 AM
To: Dana Kim
Subject: RE: plan

Middle reply.

-----Original Message-----
From: Dana Kim
Sent: Sunday, May 06, 2001 8:00 PM
To: Lee Park
Subject: plan

Original plan."""

LOTUS_MEMO = """Thanks, will do.

Dana


\x20\x20\x20
\tEnron North America Corp.
\t
\tFrom:  Pat Rowe                           09/27/2000 12:22 PM
\t

To: dana.kim@enron.com
cc: Lee Park/HOU/ECT@ECT
Subject: analysis

Here is the analysis you asked for."""

ON_WROTE = """Yes, works for me.

On Tue, 8 May 2001, Pat Rowe wrote:
> Can we move the call to 3pm?
> Thanks"""

INTERLEAVED = """> Did the contract go out?
Yes, this morning.
> And the invoice?
Tomorrow."""


def test_outlook_reply_splits_at_the_separator() -> None:
    split = split_quoted(OUTLOOK_REPLY)
    assert split.own == "Sounds good, see you Thursday.\n\nDana"
    assert len(split.quoted) == 1
    assert split.quoted[0].header.startswith("-----Original Message-----")
    assert split.quoted[0].header.endswith("Subject:\tlunch")
    assert split.quoted[0].text == "Are we still on for lunch?"


def test_lotus_forward_and_its_header_are_one_block() -> None:
    split = split_quoted(LOTUS_FORWARD)
    assert split.own == "FYI, see the note below."
    assert len(split.quoted) == 1
    assert "Pat Rowe@ENRON" in split.quoted[0].header
    assert "Adams/HOU/ECT@ECT" in split.quoted[0].header  # a wrapped To: list stays header
    assert split.quoted[0].text == "The storage meeting moves to room 3270."


@pytest.mark.parametrize(
    "forward",
    [
        "---------- Forwarded by Dana Kim/HOU/ECT on 10/16/2000 01:42\nPM ----------",
        "---------- Forwarded by Dana Kim/HOU/ECT on 10/16/2000 01:42 PM\n----------",
        "---------- Forwarded by Dana Kim/HOU/ECT on 16/10/2000 13:42\n----------",
    ],
)
def test_wrapped_forward_line_stays_header(forward: str) -> None:
    split = split_quoted(f"FYI\n\n{forward}\n\nThe memo.")
    assert split.quoted[0].text == "The memo."


def test_wrapped_forward_merges_with_the_lotus_header_below() -> None:
    body = "FYI\n\n----- Forwarded by Dana Kim/HOU/ECT on 10/16/2000 01:42 PM\n-----\n\n\nPat Rowe@ENRON\n10/12/2000 01:12 PM\nTo: Dana Kim/HOU/ECT@ECT\ncc:\nSubject: memo\n\nThe memo."
    assert [block.text for block in split_quoted(body).quoted] == ["The memo."]


def test_lotus_header_with_sender_and_date_on_one_line() -> None:
    split = split_quoted(LOTUS_INLINE_FROM)
    assert split.own == "Agreed."
    assert split.quoted[0].text == "Please confirm the collar terms."


def test_lotus_date_with_a_time_zone_still_splits() -> None:
    body = "Thanks again.\n\nFrom:\tPat Rowe/ENRON@enronXgate on 10/30/2001 05:23 PM CST\nTo:\tDana Kim/HOU/ECT@ECT\ncc:\t\n\nSubject:\tRE: Hi\n\nNo news yet."
    split = split_quoted(body)
    assert split.own == "Thanks again."
    assert split.quoted[0].text == "No news yet."


def test_lotus_memo_header_takes_its_indented_company_line() -> None:
    split = split_quoted(LOTUS_MEMO)
    assert split.own == "Thanks, will do.\n\nDana"
    assert split.quoted[0].header.startswith("Enron North America Corp.")
    assert split.quoted[0].text == "Here is the analysis you asked for."


def test_dateless_lotus_header_still_splits() -> None:
    body = "Can I use this?\n\nPaul\n\n\nTo: Dana Kim/HOU/ECT@ECT, Lee\nPark/HOU/ECT@ECT\ncc:\nSubject: Re: prices\n\nThe prices are attached."
    split = split_quoted(body)
    assert split.own == "Can I use this?\n\nPaul"
    assert split.quoted[0].text == "The prices are attached."


def test_nested_replies_come_out_outermost_first() -> None:
    split = split_quoted(NESTED)
    assert split.own == "Top reply."
    assert [block.text for block in split.quoted] == ["Middle reply.", "Original plan."]


def test_on_wrote_block_loses_its_quote_marks() -> None:
    split = split_quoted(ON_WROTE)
    assert split.own == "Yes, works for me."
    assert split.quoted[0].header.startswith("On Tue")
    assert split.quoted[0].text == "Can we move the call to 3pm?\nThanks"


def test_interleaved_quotes_leave_the_answers_as_own_text() -> None:
    split = split_quoted(INTERLEAVED)
    assert split.own == "Yes, this morning.\nTomorrow."
    assert split.quoted[0].header == ""
    assert split.quoted[0].text == "Did the contract go out?\nAnd the invoice?"


def test_a_date_in_a_sentence_is_not_a_header() -> None:
    body = "The call is on 10/12/2000 01:12 PM\nsharp, please be on time."
    assert split_quoted(body).quoted == ()


def test_normalize_strips_markup_and_mime() -> None:
    body = (
        "This is a multi-part message in MIME format.\n"
        "------_=_NextPart_001_01C1AB43.B2238F60\n"
        "Content-Type: text/plain;\n\tcharset=iso-8859-1\n"
        "<p>R&amp;D budget<br>is final</p>\n" + ("QUJD" * 20 + "\n") * 4
    )
    assert normalize(body).strip() == "R&D budget\nis final"


def test_normalize_drops_what_a_browser_hides() -> None:
    body = (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN"\n  "http://www.w3.org/TR/html4/loose.dtd">\n'
        '<style type="text/css">\nA:link {color: gray}\n</style>\n'
        "<!--\nIf you are reading this, switch to the text version.\n-->\n"
        "<script>track();</SCRIPT>Deals closed: 4 <!-- VW --> new"
    )
    assert normalize(body).strip() == "Deals closed: 4  new"


def test_a_less_than_sign_in_prose_survives() -> None:
    assert normalize("price < 3 and <style is fine") == "price < 3 and <style is fine"


def test_boilerplate_trims_the_trailing_signature_only() -> None:
    signature = "Dana Kim\nEnron North America Corp.\nPhone 713-555-0100"
    own = f"Enron North America Corp. will file today.\n\nThanks,\n{signature}"
    frequent = _frequent([signature] * 3 + ["Thanks,"] * 3, min_emails=3)
    assert strip_boilerplate(own, frequent) == "Enron North America Corp. will file today."


def test_boilerplate_never_empties_a_form_letter() -> None:
    letter = "To Mr. Ken Lay,\nPlease protect our pensions."
    frequent = _frequent([letter] * 5, min_emails=5)
    assert strip_boilerplate(letter, frequent) == letter


def test_frequent_lines_counts_emails_not_occurrences() -> None:
    assert _frequent(["a\na\na", "b"], min_emails=2) == frozenset()
    assert _frequent(["a  b", "a b"], min_emails=2) == {"a b"}


def test_a_line_is_frequent_from_the_day_its_last_needed_email_was_sent() -> None:
    dated = [(date(2001, 9, 1), "Dana"), (date(2001, 9, 2), "x"), (date(2001, 9, 3), "Dana")]
    assert frequent_lines(dated, min_emails=2) == {"Dana": date(2001, 9, 3)}


def _frequent(texts: list[str], min_emails: int) -> frozenset[str]:
    return frozenset(frequent_lines(((date(2001, 9, 1), t) for t in texts), min_emails))


## Property: build an email from known parts, split it, get the parts back.

_words = st.text(alphabet="abcdefghij KLMNO.,!?'", min_size=1, max_size=40).filter(str.strip)
_paragraph = st.lists(_words, min_size=1, max_size=4).map(
    lambda ws: "\n".join(w.strip() for w in ws)
)
_HEADERS = [
    "-----Original Message-----\nFrom: A\nSent: Monday, May 07, 2001 9:00 AM\nTo: B\nSubject: s",
    "---------------------- Forwarded by A/HOU/ECT on 10/16/2000 01:42 PM ---------------------------\n\n\nB@ENRON\n10/12/2000 01:12 PM\nTo: C/HOU/ECT@ECT\ncc:\nSubject: s",
    '"Rowe, Pat" <pat@example.com> on 10/12/2000 01:12:21 PM\nTo: D/HOU/ECT@ECT\ncc:\nSubject: s',
    "On Tue, 8 May 2001, Pat Rowe wrote:",
]


@settings(max_examples=200)
@given(
    own=_paragraph, quoted=st.lists(st.tuples(st.sampled_from(_HEADERS), _paragraph), max_size=3)
)
def test_split_recovers_the_parts_it_was_built_from(
    own: str, quoted: list[tuple[str, str]]
) -> None:
    body = own + "".join(f"\n\n{header}\n\n{text}" for header, text in quoted)
    split = split_quoted(body)
    assert split.own == own.strip()
    assert [block.text for block in split.quoted] == [text.strip() for _, text in quoted]


@settings(max_examples=200)
@given(
    own=_paragraph, quoted=st.lists(st.tuples(st.sampled_from(_HEADERS), _paragraph), max_size=3)
)
def test_splitting_own_text_again_changes_nothing(own: str, quoted: list[tuple[str, str]]) -> None:
    body = own + "".join(f"\n\n{header}\n\n{text}" for header, text in quoted)
    once = split_quoted(body).own
    assert split_quoted(once).own == once


@settings(max_examples=200)
@given(lines=st.lists(_words, min_size=1, max_size=8), frequent=st.frozensets(_words, max_size=6))
def test_boilerplate_keeps_a_prefix_and_never_returns_nothing(
    lines: list[str], frequent: frozenset[str]
) -> None:
    own = "\n".join(lines)
    keys = frozenset(" ".join(line.split()) for line in frequent)
    trimmed = strip_boilerplate(own, keys)
    assert own.startswith(trimmed)
    assert trimmed.strip()


## Who wrote a quoted message, in each header layout.


@pytest.mark.parametrize(
    ("header", "author"),
    [
        ("From: Pat Rowe on 07/05/2001 02:28 PM\nTo: Dana Kim/NA/Enron@Enron", "Pat Rowe"),
        ("Lee Park\n\t07/05/2001 03:32 PM\n\t\t To: Dana Kim/NA/Enron@Enron", "Lee Park"),
        ("Robin Cole/ENRON@enronXgate\n\t07/05/2001 02:20 PM", "Robin Cole"),
        ("Sam Bell@ECT\n\t01/05/2001 08:44 AM", "Sam Bell"),
        (
            '"Kit Moss" <KMOSS@LAW.COM> on 01/04/2001 06:01:45 PM\nTo: <pat.rowe@enron.com>',
            "Kit Moss",
        ),
        (
            "-----Original Message-----\nFrom: \tPat Rowe [mailto:pat@x.com]\nSent: Monday",
            "Pat Rowe",
        ),
        (
            "----- Forwarded by Lee Park/NA/Enron on 07/05/2001 02:27 PM -----\n\nPat Rowe",
            "Pat Rowe",
        ),
        ("From: pat.rowe@enron.com\nTo: dana.kim@enron.com", "pat.rowe@enron.com"),
        ("To: Dana Kim/HOU/EES@EES\ncc:\nSubject: lunch", ""),
        ("", ""),
    ],
)
def test_block_author(header: str, author: str) -> None:
    assert block_author(header) == author


def test_block_subject_reads_the_field_wherever_it_sits() -> None:
    assert block_subject("Lee Park\n\t\t To: Dana Kim\n\t\t Subject: Re: Dunn Subpoena") == (
        "Re: Dunn Subpoena"
    )
    assert block_subject("On Monday, Pat wrote:") == ""
