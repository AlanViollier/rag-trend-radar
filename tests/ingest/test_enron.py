from pathlib import Path

import pytest

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.records import RawFile
from rag_trend_radar.sources.enron import parse, paths, read_raw_file

HEADERS = (
    "Message-ID: <1.2.JavaMail.evans@thyme>\r\n"
    "Date: Fri, 25 May 2001 14:37:24 -0700 (PDT)\r\n"
    "From: dana.kim@enron.com\r\n"
    "To: pat.rowe@enron.com, Lee Park <LEE.PARK@enron.com>,\r\n\tpat.rowe@enron.com\r\n"
)


def _raw(content: bytes) -> RawFile:
    return RawFile(path="kim-d/inbox/1.", mailbox="kim-d", folder="inbox", content=content)


def test_parse_decodes_headers_and_quoted_printable_body() -> None:
    raw = _raw(
        (
            HEADERS + "Subject: =?cp1252?Q?=93Grade=94?=\r\n"
            "Content-Type: text/plain; charset=us-ascii\r\n"
            "Content-Transfer-Encoding: quoted-printable\r\n\r\n"
            "The line of credit is=\r\n actually $112k.\r\n=20\r\n"
        ).encode()
    )
    email = parse(raw)
    assert email.to_addrs == ("pat.rowe@enron.com", "lee.park@enron.com")
    assert email.subject == "“Grade”"
    assert email.body == "The line of credit is actually $112k.\n \n"
    assert email.sent_at.utcoffset() is not None


def test_mislabelled_bytes_fall_back_to_cp1252() -> None:
    raw = _raw((HEADERS + "Subject: =??Q?=93A=94?=\r\n\r\n").encode() + b"it\x92s done\r\n")
    email = parse(raw)
    assert email.subject == "“A”"
    assert email.body == "it’s done\n"


@pytest.mark.parametrize(
    ("headers", "body", "reason"),
    [
        (
            "Date: Mon, 31 Dec 1979 16:00:00 -0800 (PST)\r\nFrom: a@enron.com\r\n",
            "x",
            "date outside",
        ),
        ("Date: someday\r\nFrom: a@enron.com\r\n", "x", "unparseable date"),
        (HEADERS.replace("dana.kim@", "postmaster@"), "x", "bounce"),
        (HEADERS + "Subject: Undeliverable: lunch\r\n", "x", "bounce"),
        (HEADERS, "<OMNI>\r\n<OMNIPAB>PERSON~Dana", "address book dump"),
    ],
)
def test_unusable_files_raise_with_a_reason(headers: str, body: str, reason: str) -> None:
    raw = _raw(f"Message-ID: <x>\r\n{headers}\r\n{body}".encode())
    with pytest.raises(ParseError, match=reason):
        parse(raw)


def test_files_are_read_with_their_mailbox_and_nested_folder(tmp_path: Path) -> None:
    (tmp_path / "kim-d" / "projects" / "west").mkdir(parents=True)
    (tmp_path / "kim-d" / "projects" / "west" / "3.").write_bytes(b"x")
    (tmp_path / "kim-d" / "1.").write_bytes(b"y")
    files = [read_raw_file(tmp_path, path) for path in paths(tmp_path)]
    assert [(f.path, f.mailbox, f.folder) for f in files] == [
        ("kim-d/1.", "kim-d", ""),
        ("kim-d/projects/west/3.", "kim-d", "projects/west"),
    ]


def test_a_folded_subject_is_unfolded() -> None:
    raw = _raw(
        (HEADERS + "Subject: Global Employment Law Alert for\r\n \tMid-September\r\n\r\nx").encode()
    )
    assert parse(raw).subject == "Global Employment Law Alert for Mid-September"


def test_an_address_missing_its_first_name_falls_back_to_the_name_shown() -> None:
    raw = _raw(
        b"Message-ID: <3.4.JavaMail.evans@thyme>\r\n"
        b"Date: Wed, 30 May 2001 08:19:41 -0700 (PDT)\r\n"
        b"From: trading <.williams@enron.com>\r\n"
        b"Subject: RE: Numbers\r\n"
        b"X-From: Williams, Jason (Trading) </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JWILLIA>\r\n\r\n"
        b"Numbers attached.\r\n"
    )
    assert parse(raw).sender == "Williams, Jason (Trading)"


def test_a_whole_address_stays_as_it_is() -> None:
    raw = _raw((HEADERS + "Subject: x\r\nX-From: Kim, Dana\r\n\r\nbody\r\n").encode())
    assert parse(raw).sender == "dana.kim@enron.com"
