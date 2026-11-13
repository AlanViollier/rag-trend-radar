"""The agent's tools see what their person may see, as of the day asked, and nothing more.

Two mailboxes: kim-d holds a secret about the quokka ledger, rowe-p holds three notes about it,
one sent after the day asked. The radar has one topic with an alert.
"""

from collections.abc import Iterator
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from rag_trend_radar import index
from rag_trend_radar.agent.desk import Desk, ToolResult
from rag_trend_radar.agent.tools import call, definitions
from rag_trend_radar.agent.tools import search as search_tool
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.exceptions import ToolRefused
from rag_trend_radar.records import Document, EmailHeaders, SearchHit

MODEL, DIMS = "test/agent", 8
CHICAGO = ZoneInfo("America/Chicago")
AS_OF = date(2001, 10, 31)
HIDDEN = "The quokka ledger is short by forty million."
LATER = datetime(2001, 11, 20, 9, 0, tzinfo=CHICAGO)


def _email(external_id: str, body: str, mailbox: str, sender: str, sent_at: datetime) -> Document:
    return Document(
        source="agent-test",
        external_id=external_id,
        kind="email",
        sent_at=sent_at,
        title=f"Note {external_id}",
        author=sender,
        body=body,
        quoted="",
        readers=(mailbox,),
        email=EmailHeaders(
            message_id=external_id,
            sender=sender,
            to_addrs=("team@enron.com",),
            cc_addrs=(),
            bcc_addrs=(),
        ),
    )


@pytest.fixture(scope="module")
def world(store: None) -> Iterator[dict[str, int]]:
    october = datetime(2001, 10, 10, 9, 0, tzinfo=CHICAGO)
    documents = [
        _email("secret", HIDDEN, "kim-d", "vince.kaminski@enron.com", october),
        *(
            _email(f"rowe-{n}", f"Quokka ledger note {n}.", "rowe-p", "rowe@enron.com", october)
            for n in range(2)
        ),
        _email("rowe-late", "Quokka ledger, the last note.", "rowe-p", "rowe@enron.com", LATER),
        _email(
            "nicolay",
            "Quokka ledger from a lookalike name.",
            "rowe-p",
            "c.nicolay@enron.com",
            october,
        ),
    ]
    with db.connect() as conn:
        conn.cursor().executemany(
            "INSERT INTO employees (mailbox) VALUES (%s) ON CONFLICT DO NOTHING",
            [("kim-d",), ("rowe-p",), ("kaminski-v",)],
        )
        index.upsert_documents(conn, documents, lambda doc: [f"Subject: {doc.title}\n\n{doc.body}"])
        rows = conn.execute(
            "SELECT c.id, d.external_id, d.id FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.source = 'agent-test'"
        ).fetchall()
        index.write_vectors(
            conn, MODEL, [i for i, _, _ in rows], np.eye(DIMS, dtype=np.float32)[: len(rows)]
        )
        conn.execute(
            "INSERT INTO clusters (id, model, first_week, last_week, kind, shown)"
            " VALUES (900001, 'test', '2001-10-07', '2001-10-28', 'topic', true)"
        )
        for week, recent in (("2001-10-07", 2), ("2001-10-28", 40)):
            conn.execute(
                "INSERT INTO cluster_weeks VALUES (900001, %s, 0, '[1,0,0,0]', '{quokka,ledger}',"
                " %s, %s, %s, 0, 5.0, 0.0, true)",
                (week, recent, recent, recent),
            )
        conn.execute(
            "INSERT INTO alerts (day, cluster_id, observed, expected) VALUES ('2001-10-28', 900001, 40, 2)"
        )
    yield {external_id: document_id for _, external_id, document_id in rows}
    db.reset_radar()
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'agent-test'")


def _run(person: str, tool: str, arguments: dict[str, object]) -> ToolResult:
    with db.connect(person) as conn:
        return call(Desk(conn, person, AS_OF), tool, arguments)


def _text(result: ToolResult) -> str:
    return "\n".join(str(block.get("text", "")) for block in result.blocks)


@pytest.fixture
def no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search without loading the real embedding model or reranker: one fixed query vector."""
    settings = get_settings().model_copy(update={"embedding_model": MODEL, "embedding_dims": DIMS})
    monkeypatch.setattr(search_tool, "get_settings", lambda: settings)
    monkeypatch.setattr("rag_trend_radar.search.pipeline.get_settings", lambda: settings)
    monkeypatch.setattr(
        search_tool, "embed", lambda texts, *_, **__: np.ones((1, DIMS), np.float32)
    )

    def unchanged(_c: object, _q: str, hits: list[SearchHit], _m: str, top: int) -> list[SearchHit]:
        return [hit.model_copy(update={"score": 1.0}) for hit in hits[:top]]

    monkeypatch.setattr("rag_trend_radar.search.pipeline.rerank", unchanged)


@pytest.mark.usefixtures("world", "no_models")
def test_an_employee_searches_only_their_own_mail_and_only_up_to_the_day() -> None:
    rowe = _run("rowe-p", "search", {"query": "quokka ledger"})
    assert HIDDEN not in _text(rowe)
    assert len(rowe.documents) == 3
    assert "last note" not in str(rowe.blocks)
    kim = _run("kim-d", "search", {"query": "quokka ledger"})
    assert HIDDEN in str(kim.blocks)


@pytest.mark.usefixtures("world", "no_models")
def test_search_dates_narrow_within_the_day_asked() -> None:
    assert _run("investigator", "search", {"query": "quokka", "until": "2002-01-31"}).documents
    later = _run(
        "investigator", "search", {"query": "quokka", "since": "2001-11-01", "until": "2002-01-31"}
    )
    assert later.documents == ()


@pytest.mark.usefixtures("world")
def test_counts_hold_only_the_persons_mail_up_to_the_day() -> None:
    assert _run(
        "rowe-p", "mail_counts", {"words": "quokka ledger", "by": "total"}
    ).summary.startswith("3 emails")
    assert _run(
        "kim-d", "mail_counts", {"words": "quokka ledger", "by": "total"}
    ).summary.startswith("1 emails")
    everyone = _run(
        "investigator", "mail_counts", {"words": "quokka", "until": "2002-12-31", "by": "total"}
    )
    assert everyone.summary.startswith("4 emails")


@pytest.mark.usefixtures("world")
def test_reading_an_email_needs_the_right_to_read_it_by_then(world: dict[str, int]) -> None:
    assert _run("rowe-p", "read_email", {"document_id": world["secret"]}).summary == "not readable"
    assert HIDDEN in str(_run("kim-d", "read_email", {"document_id": world["secret"]}).blocks)
    assert (
        _run("investigator", "read_email", {"document_id": world["rowe-late"]}).summary
        == "not readable"
    )


@pytest.mark.usefixtures("world")
def test_a_name_finds_its_mailbox_and_whole_name_addresses_first() -> None:
    found = _text(_run("investigator", "find_person", {"name": "Vince Kaminski"}))
    assert "kaminski-v" in found
    assert "vince.kaminski@enron.com" in found
    lines = _text(_run("investigator", "find_person", {"name": "Rowe"})).splitlines()
    assert lines[2].startswith("rowe@enron.com")


@pytest.mark.usefixtures("world")
def test_the_radar_is_for_the_investigator_only() -> None:
    with db.connect("rowe-p") as conn:
        desk = Desk(conn, "rowe-p", AS_OF)
        assert {d["name"] for d in definitions(desk)} == {
            "search",
            "read_email",
            "find_person",
            "mail_counts",
        }
        for tool in ("top_topics", "topic_history", "alerts"):
            with pytest.raises(ToolRefused):
                call(desk, tool, {})
        # And if a tool were offered by mistake, row-level security still gives it nothing.
        assert conn.execute("SELECT count(*) FROM cluster_weeks").fetchall() == [(0,)]


@pytest.mark.usefixtures("world")
def test_the_investigator_reads_topics_and_alerts() -> None:
    top = _text(_run("investigator", "top_topics", {"since": "2001-10-08", "words": "quokka"}))
    assert "topic 900001" in top
    assert "2 → 40" in top
    assert "topic 900001" in _text(_run("investigator", "alerts", {"since": "2001-10-01"}))
    assert "2001-10-28  40" in _text(_run("investigator", "topic_history", {"topic": 900001}))


@pytest.mark.usefixtures("world")
def test_arguments_outside_the_schema_are_refused() -> None:
    with pytest.raises(ToolRefused):
        _run("investigator", "mail_counts", {"words": "quokka", "sql": "DROP TABLE documents"})
    with pytest.raises(ToolRefused):
        _run("investigator", "search", {})


@pytest.mark.usefixtures("world", "no_models")
def test_a_run_never_gets_the_same_chunk_twice(world: dict[str, int]) -> None:
    with db.connect("rowe-p") as conn:
        desk = Desk(conn, "rowe-p", AS_OF)
        first = call(desk, "search", {"query": "quokka ledger"})
        again = call(desk, "search", {"query": "quokka ledger"})
    assert len(first.documents) == 3
    assert again.documents == ()
    assert again.summary == "0 new results"
    assert all(str(d) in again.aside or str(d) in _text(again) for d in first.documents)


@pytest.mark.usefixtures("world", "no_models")
def test_an_email_read_whole_leaves_later_searches(world: dict[str, int]) -> None:
    with db.connect("rowe-p") as conn:
        desk = Desk(conn, "rowe-p", AS_OF)
        call(desk, "read_email", {"document_id": world["rowe-0"]})
        found = call(desk, "search", {"query": "quokka ledger"})
    assert world["rowe-0"] not in found.documents
    assert len(found.documents) == 2
    assert str(world["rowe-0"]) in found.aside


@pytest.mark.usefixtures("world")
def test_addresses_may_come_as_one_comma_separated_string() -> None:
    listed = _run("investigator", "mail_counts", {"senders": ["rowe@enron.com"], "by": "total"})
    joined = _run(
        "investigator", "mail_counts", {"senders": "rowe@enron.com, x@y.com", "by": "total"}
    )
    assert listed.summary == joined.summary


@pytest.mark.usefixtures("world")
def test_a_lookup_passes_when_a_cited_email_is_a_labelled_one(world: dict[str, int]) -> None:
    from rag_trend_radar.evals import agent as checks
    from rag_trend_radar.records import AgentQuestion, AgentRun, Citation

    question = AgentQuestion(
        id="l", kind="lookup", question="Q", person="rowe-p", relevant=("rowe-0",)
    )

    def cites(document: int) -> AgentRun:
        return AgentRun(
            question="Q", person="rowe-p", as_of=AS_OF, model="m", prompt_version=5, steps=(),
            answer="A", found="yes", citations=(Citation(document_id=document, cited_text="x"),),
            cost_usd=0, seconds=0, started=datetime(2026, 11, 10, tzinfo=CHICAGO), commit="c",
        )  # fmt: skip

    with db.connect() as conn:
        assert checks.score(conn, question, cites(world["rowe-0"])).passed
        assert not checks.score(conn, question, cites(world["rowe-late"])).passed


@pytest.mark.usefixtures("world")
def test_several_phrases_count_each_email_once() -> None:
    def total(words: list[str]) -> str:
        return _run("investigator", "mail_counts", {"words": words, "by": "total"}).summary

    assert total(["lookalike", "note 0"]).startswith("2 emails")
    # Every email holds both words: counted once each, not twice.
    assert total(["quokka", "ledger"]).startswith("4 emails")


@pytest.mark.usefixtures("world")
def test_a_phrase_with_small_words_still_counts() -> None:
    # "from a" are words the index drops: as a plain phrase this matched nothing.
    found = _run(
        "investigator", "mail_counts", {"words": ["ledger from a lookalike"], "by": "total"}
    )
    assert found.summary.startswith("1 emails")


@pytest.mark.usefixtures("world")
def test_a_one_week_range_grows_from_the_last_week_before_it() -> None:
    # The topic has weeks ending Oct 7 (2) and Oct 28 (40) only: Oct 28 alone grows from Oct 7.
    one_week = _text(
        _run("investigator", "top_topics", {"since": "2001-10-22", "until": "2001-10-28"})
    )
    assert "2 → 40 (+38)" in one_week
