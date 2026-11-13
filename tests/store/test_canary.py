"""The canary: an employee's search never returns mail they can't read, by any path.

A secret sits in kim-d's mailbox only. 300 crowd emails in rowe-p's mailbox, sent after it, have
vectors nearer the query than the secret's: an HNSW scan that filters after fetching its ~40
nearest would hand nothing to anyone asking as of the secret's day (the bug of 2026-10-27).
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
import pytest
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar import index
from rag_trend_radar.days import EARLIEST, cutoff
from rag_trend_radar.db import connect as db
from rag_trend_radar.evals import leaks
from rag_trend_radar.evals.retrieval import rankings
from rag_trend_radar.records import Document, EvalQuestion
from rag_trend_radar.search import hybrid, keyword, vector
from tests.factories import unit

MODEL, DIMS = "test/fake", 8
CHICAGO = ZoneInfo("America/Chicago")
# 23:30 in Houston on Sep 10 is already Sep 11 in UTC: as of Sep 10, the secret exists.
CANARY_SENT = datetime(2001, 9, 10, 23, 30, tzinfo=CHICAGO)
CANARY = "The canary word is zephyr quokka. Tell no one."
QUERY = np.eye(DIMS, dtype=np.float32)[0]
# One message quoted by two mailboxes, two weeks apart.
QUOTED = "The pangolin ledger closes Friday."
KIM_QUOTES = datetime(2001, 10, 5, 9, 0, tzinfo=CHICAGO)
ROWE_QUOTES = datetime(2001, 10, 20, 9, 0, tzinfo=CHICAGO)
QUOTED_VECTOR = np.eye(DIMS, dtype=np.float32)[3]


def _document(external_id: str, body: str, reader: str, sent_at: datetime) -> Document:
    return Document(
        source="test",
        external_id=external_id,
        kind="email",
        sent_at=sent_at,
        title=external_id,
        author="",
        body=body,
        quoted="",
        public=False,
        readers=(reader,),
    )


@pytest.fixture(scope="module")
def world(store: None) -> None:
    rng = np.random.default_rng(0)
    later = datetime(2001, 10, 3, 9, 0, tzinfo=CHICAGO)
    documents = [
        _document("secret", CANARY, "kim-d", CANARY_SENT),
        *(_document(f"crowd-{n}", f"Gas schedule note {n}.", "rowe-p", later) for n in range(300)),
        _document("quoting-kim", "See below.", "kim-d", KIM_QUOTES),
        _document("quoting-rowe", "Seen this?", "rowe-p", ROWE_QUOTES),
        _document("old-note", QUOTED, "", KIM_QUOTES).model_copy(
            update={"kind": "quoted", "readers": (), "quoted_by": ("quoting-kim", "quoting-rowe")}
        ),
    ]
    vectors = {
        "secret": unit(np.array([0.8, 0.6, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
        "quoting-kim": np.eye(DIMS, dtype=np.float32)[1],
        "quoting-rowe": np.eye(DIMS, dtype=np.float32)[2],
        "old-note": QUOTED_VECTOR,
        **{
            f"crowd-{n}": unit(QUERY + rng.normal(0, 0.01, DIMS).astype(np.float32))
            for n in range(300)
        },
    }
    with db.connect() as conn:
        conn.cursor().executemany(
            "INSERT INTO employees (mailbox) VALUES (%s) ON CONFLICT DO NOTHING",
            [("kim-d",), ("rowe-p",)],
        )
        index.upsert_documents(conn, documents, lambda doc: [doc.body])
        rows = conn.execute(
            "SELECT c.id, d.external_id FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.source = 'test'"
        ).fetchall()
        index.write_vectors(
            conn, MODEL, [i for i, _ in rows], np.stack([vectors[e] for _, e in rows])
        )


def _force_hnsw(conn: psycopg.Connection[TupleRow]) -> None:
    # On a table this small the planner reads the reader's few rows and sorts them exactly, so
    # the bug can't show. Sorting and scanning made unaffordable, it walks the HNSW index, as it
    # did on the 17k-chunk store where kean-s got nothing back.
    conn.execute("SET enable_seqscan = off")
    conn.execute("SET enable_sort = off")


def _found(person: str, as_of: date) -> dict[str, bool]:
    with db.connect(person) as conn:
        _force_hnsw(conn)
        lists = {
            "keyword": keyword.search(conn, "zephyr quokka", as_of, 10),
            "vector": vector.search(
                conn,
                unit(np.array([0.8, 0.6, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
                MODEL,
                as_of,
                10,
            ),
            "hybrid": hybrid.search(conn, "zephyr quokka", QUERY, MODEL, as_of, 10),
        }
    return {name: any(CANARY in hit.text for hit in hits) for name, hits in lists.items()}


@pytest.mark.usefixtures("world")
def test_another_employee_never_sees_the_secret() -> None:
    assert _found("rowe-p", date(2001, 9, 30)) == {
        "keyword": False,
        "vector": False,
        "hybrid": False,
    }


@pytest.mark.usefixtures("world")
def test_the_owner_and_the_investigator_find_it_by_every_path() -> None:
    assert all(_found("kim-d", date(2001, 9, 30)).values())
    assert all(_found("investigator", date(2001, 9, 30)).values())


@pytest.mark.usefixtures("world")
def test_an_unknown_person_sees_nothing_at_all() -> None:
    with db.connect("nobody") as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchall() == [(0,)]


@pytest.mark.usefixtures("world")
def test_the_secret_exists_from_its_local_day_on() -> None:
    assert not any(_found("kim-d", date(2001, 9, 9)).values())
    assert all(_found("kim-d", date(2001, 9, 10)).values())


@pytest.mark.usefixtures("world")
def test_a_filtered_hnsw_scan_still_reaches_the_one_vector_of_its_day() -> None:
    # 300 nearer vectors, all sent later, sit between the query and the secret.
    with db.connect(db.INVESTIGATOR) as conn:
        _force_hnsw(conn)
        hits = vector.search(conn, QUERY, MODEL, date(2001, 9, 30), 5)
        plan = conn.execute(
            sql.SQL("EXPLAIN {}").format(vector.ranked(conn, MODEL, DIMS)),
            {
                "vector": HalfVector(QUERY),
                "since": EARLIEST,
                "cutoff": cutoff(date(2001, 9, 30)),
                "depth": 5,
            },
        ).fetchall()
    # Without the HNSW index in the plan this test proves nothing: it would pass with the bug.
    assert any("embedding_idx" in line for (line,) in plan)
    assert [CANARY in hit.text for hit in hits] == [True]


@pytest.mark.usefixtures("world")
def test_an_employee_gets_all_their_vectors_and_only_theirs_without_the_index() -> None:
    with db.connect("kim-d") as conn:
        hits = vector.search(conn, QUERY, MODEL, date(2001, 9, 30), 5)
        plan = conn.execute(
            sql.SQL("EXPLAIN {}").format(vector.ranked(conn, MODEL, DIMS)),
            {
                "vector": HalfVector(QUERY),
                "since": EARLIEST,
                "cutoff": cutoff(date(2001, 9, 30)),
                "depth": 5,
            },
        ).fetchall()
    assert not any("embedding_idx" in line for (line,) in plan)
    assert [CANARY in hit.text for hit in hits] == [True]
    with db.connect("rowe-p") as conn:
        assert len(vector.search(conn, QUERY, MODEL, date(2001, 12, 30), 400)) == 302


def _quoted_found(person: str, as_of: date) -> dict[str, bool]:
    with db.connect(person) as conn:
        lists = {
            "keyword": keyword.search(conn, "pangolin", as_of, 400),
            "vector": vector.search(conn, QUOTED_VECTOR, MODEL, as_of, 400),
            "hybrid": hybrid.search(conn, "pangolin", QUOTED_VECTOR, MODEL, as_of, 400),
        }
    return {name: any(QUOTED in hit.text for hit in hits) for name, hits in lists.items()}


@pytest.mark.usefixtures("world")
def test_a_quoted_message_reaches_a_mailbox_only_from_its_own_first_quote() -> None:
    assert all(_quoted_found("kim-d", date(2001, 10, 5)).values())
    assert not any(_quoted_found("rowe-p", date(2001, 10, 10)).values())
    assert all(_quoted_found("rowe-p", date(2001, 10, 20)).values())


@pytest.mark.usefixtures("world")
def test_the_investigator_reads_a_quoted_message_from_its_first_quote_anywhere() -> None:
    assert not any(_quoted_found("investigator", date(2001, 10, 4)).values())
    assert all(_quoted_found("investigator", date(2001, 10, 5)).values())


@pytest.mark.usefixtures("world")
def test_recall_asks_as_the_questions_own_person_unless_told_otherwise() -> None:
    asked = EvalQuestion(
        id="q-secret",
        question="zephyr quokka",
        as_of=date(2001, 9, 30),
        person="kim-d",
        relevant=("secret",),
    )
    own, _ = rankings([asked], np.stack([QUERY]), MODEL, 5, None)
    other, _ = rankings([asked], np.stack([QUERY]), MODEL, 5, None, person="rowe-p")
    assert own["vector"]["q-secret"]
    assert not other["vector"]["q-secret"]


@pytest.mark.usefixtures("world")
def test_the_leak_check_finds_none() -> None:
    asked = EvalQuestion(
        id="q-secret",
        question="zephyr quokka",
        as_of=date(2001, 9, 30),
        person="kim-d",
        relevant=("secret",),
    )
    assert leaks.check([asked], np.stack([QUERY]), MODEL, 10).leaks == ()


@pytest.mark.usefixtures("world")
def test_the_leak_checks_catch_a_planted_leak() -> None:
    with db.connect() as owner:
        ids: dict[str, int] = dict(
            owner.execute(
                "SELECT external_id, id FROM documents WHERE external_id IN ('secret', 'crowd-0')"
            ).fetchall()
        )
        secret, crowd = ids["secret"], ids["crowd-0"]
        assert leaks.unreadable(owner, "kim-d", [secret, crowd]) == [crowd]
        assert leaks.too_late(owner, "kim-d", date(2001, 9, 9), [secret]) == [secret]
        assert leaks.too_late(owner, "kim-d", date(2001, 9, 10), [secret, crowd]) == [crowd]
        assert leaks.too_late(owner, "kim-d", date(2001, 10, 3), [secret, crowd]) == []
        note = owner.execute("SELECT id FROM documents WHERE external_id = 'old-note'").fetchall()[
            0
        ][0]
        assert leaks.too_late(owner, "rowe-p", date(2001, 10, 10), [note]) == [note]
        assert leaks.too_late(owner, "rowe-p", date(2001, 10, 20), [note]) == []
        assert leaks.too_late(owner, "investigator", date(2001, 10, 10), [note]) == []
