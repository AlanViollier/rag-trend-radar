from pathlib import Path

import pytest

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.schema import RawFile
from rag_trend_radar.sources.enron import iter_raw_files, parse

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


def test_iter_raw_files_reads_mailbox_and_nested_folder(tmp_path: Path) -> None:
    (tmp_path / "kim-d" / "projects" / "west").mkdir(parents=True)
    (tmp_path / "kim-d" / "projects" / "west" / "3.").write_bytes(b"x")
    (tmp_path / "kim-d" / "1.").write_bytes(b"y")
    files = list(iter_raw_files(tmp_path))
    assert [(f.path, f.mailbox, f.folder) for f in files] == [
        ("kim-d/1.", "kim-d", ""),
        ("kim-d/projects/west/3.", "kim-d", "projects/west"),
    ]


def test_a_folded_subject_is_unfolded() -> None:
    raw = _raw(
        (HEADERS + "Subject: Global Employment Law Alert for\r\n \tMid-September\r\n\r\nx").encode()
    )
    assert parse(raw).subject == "Global Employment Law Alert for Mid-September"
