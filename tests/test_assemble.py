from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rag_trend_radar.ingest.assemble import assemble
from rag_trend_radar.schema import Document

CHICAGO = ZoneInfo("America/Chicago")
QUOTE = "-----Original Message-----\nFrom: Kit Moss\nSent: Friday\nTo: Lee Park\nSubject: Gas\n\n"


def _write(root: Path, path: str, sent: str, subject: str, body: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        f"Message-ID: <{path}>\nDate: {sent}\nFrom: someone@enron.com\nTo: lee.park@enron.com\n"
        f"Subject: {subject}\n\n{body}"
    )


def _documents(root: Path) -> dict[str, Document]:
    span = assemble(root, date(2001, 9, 1), date(2001, 9, 30), CHICAGO)
    return {d.title: d for d in span.documents}


def test_copies_merge_into_one_email_readable_from_every_mailbox(tmp_path: Path) -> None:
    _write(tmp_path, "kim-d/inbox/1.", "Mon, 10 Sep 2001 09:00:00 -0500", "plan", "at 3")
    _write(tmp_path, "rowe-p/sent/7.", "Mon, 10 Sep 2001 09:00:00 -0500", "plan", "at  3\n")
    plan = _documents(tmp_path)["plan"]
    assert plan.readers == ("kim-d", "rowe-p")
    assert plan.paths == ("kim-d/inbox/1.", "rowe-p/sent/7.")


def test_the_span_is_cut_on_the_resolved_date_in_local_days(tmp_path: Path) -> None:
    ## 23:30 on Aug 31 in Chicago by one export, 06:30:12 Sep 1 by the server copy, which wins.
    _write(tmp_path, "kim-d/inbox/2.", "Fri, 31 Aug 2001 23:30 -0500", "edge", "x")
    _write(tmp_path, "rowe-p/inbox/2.", "Sat, 1 Sep 2001 06:30:12 -0500", "edge", "x")
    _write(tmp_path, "kim-d/inbox/3.", "Fri, 31 Aug 2001 22:00:00 -0500", "august", "y")
    documents = _documents(tmp_path)
    assert documents["edge"].sent_at == datetime(2001, 9, 1, 11, 30, 12, tzinfo=UTC)
    assert "august" not in documents


def test_a_quoted_message_no_email_holds_is_one_document_for_all_who_quote_it(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "kim-d/inbox/4.",
        "Mon, 3 Sep 2001 10:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Prices rose.",
    )
    _write(
        tmp_path,
        "park-l/inbox/5.",
        "Tue, 4 Sep 2001 10:00 -0500",
        "FW: Gas",
        "fyi\n\n" + QUOTE + "Prices rose.",
    )
    gas = _documents(tmp_path)["Gas"]
    assert (gas.kind, gas.author, gas.body, gas.readers) == (
        "quoted",
        "Kit Moss",
        "Prices rose.",
        ("kim-d", "park-l"),
    )
    assert gas.sent_at == datetime(2001, 9, 3, 15, 0, tzinfo=UTC)


def test_a_quote_of_an_email_we_hold_adds_no_document(tmp_path: Path) -> None:
    _write(tmp_path, "kim-d/sent/6.", "Mon, 3 Sep 2001 10:00 -0500", "Gas", "Prices rose.")
    _write(
        tmp_path,
        "park-l/inbox/6.",
        "Mon, 3 Sep 2001 11:00 -0500",
        "RE: Gas",
        "ok\n\n" + QUOTE + "Prices rose.",
    )
    assert [d.kind for d in _documents(tmp_path).values()] == ["email", "email"]


def test_junk_dates_are_quarantined_and_kept(tmp_path: Path) -> None:
    _write(tmp_path, "kim-d/inbox/8.", "Mon, 31 Dec 1979 16:00:00 -0800", "old", "z")
    span = assemble(tmp_path, date(2001, 9, 1), date(2001, 9, 30), CHICAGO)
    assert span.documents == ()
    assert [f.path for f in span.raw_files] == ["kim-d/inbox/8."]
    assert span.rejected["kim-d/inbox/8."].startswith("date outside")
