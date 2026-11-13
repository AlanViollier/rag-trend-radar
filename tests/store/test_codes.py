from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.evals.questions import code_holds
from rag_trend_radar.records import Document, EvalQuestion

SENT = datetime(2001, 9, 5, 10, 0, tzinfo=ZoneInfo("America/Chicago"))


def _document(external_id: str, body: str) -> Document:
    return Document(
        source="test",
        external_id=external_id,
        kind="email",
        sent_at=SENT,
        title=external_id,
        author="",
        body=body,
        quoted="",
        readers=("lay-k",),
    )


@pytest.fixture(scope="module")
def mailbox(store: None) -> None:
    documents = [
        _document("deal", "Please confirm deal NG-617161 by Friday."),
        _document("near-miss", "Ticket 96176161 and x617161y are other numbers."),
        *(_document(f"report-{n}", f"Daily report {n}: contract 555123 flowed.") for n in range(6)),
    ]
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('lay-k') ON CONFLICT DO NOTHING")
        index.upsert_documents(conn, documents, lambda doc: [doc.body])


def _asked(question: str, code: str, label: str) -> EvalQuestion:
    return EvalQuestion(
        id="q",
        question=question,
        as_of=date(2001, 9, 30),
        person="lay-k",
        relevant=(label,),
        code=code,
    )


@pytest.mark.usefixtures("mailbox")
@pytest.mark.parametrize(
    ("question", "code", "label", "kept"),
    [
        ("Who has to confirm deal 617161?", "617161", "deal", True),
        ("Who has to confirm deal 617161?", "", "deal", False),
        ("Who has to confirm the gas deal?", "617161", "deal", False),
        ("Which deal is 617161?", "617161", "near-miss", False),
        ("What flowed on contract 555123?", "555123", "report-0", False),
    ],
    ids=["one-holder", "no-code", "code-not-in-question", "label-lacks-token", "too-many-holders"],
)
def test_a_code_must_be_quoted_held_by_its_label_and_by_few(
    question: str, code: str, label: str, kept: bool
) -> None:
    with db.connect("lay-k") as conn:
        assert code_holds(conn, _asked(question, code, label)) is kept
