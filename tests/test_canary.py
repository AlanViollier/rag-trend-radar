"""The canary: an employee's search never returns mail they can't read, by any path.

A secret sits in kim-d's mailbox only. 300 crowd emails in rowe-p's mailbox have vectors nearer
the query than the secret's, so an HNSW scan that filters after fetching its ~40 nearest would
hand kim-d nothing (the bug of 2026-10-27).
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
import pytest
from numpy.typing import NDArray
from pgvector import HalfVector
from psycopg import sql
from psycopg.rows import TupleRow

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import Assembled, Document
from rag_trend_radar.search import hybrid, keyword, vector
from rag_trend_radar.search.ranking import cutoff

MODEL, DIMS = "test/fake", 8
CHICAGO = ZoneInfo("America/Chicago")
## 23:30 in Houston on Sep 10 is already Sep 11 in UTC: as of Sep 10, the secret exists.
CANARY_SENT = datetime(2001, 9, 10, 23, 30, tzinfo=CHICAGO)
CANARY = "The canary word is zephyr quokka. Tell no one."
QUERY = np.eye(DIMS, dtype=np.float32)[0]


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
        body_hash=external_id,
        public=False,
        readers=(reader,),
    )


def _unit(v: NDArray[np.float32]) -> NDArray[np.float32]:
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture(scope="module")
def world(store: None) -> None:
    rng = np.random.default_rng(0)
    early = datetime(2001, 9, 3, 9, 0, tzinfo=CHICAGO)
    documents = [
        _document("secret", CANARY, "kim-d", CANARY_SENT),
        *(_document(f"crowd-{n}", f"Gas schedule note {n}.", "rowe-p", early) for n in range(300)),
    ]
    vectors = {
        "secret": _unit(np.array([0.8, 0.6, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
        **{
            f"crowd-{n}": _unit(QUERY + rng.normal(0, 0.01, DIMS).astype(np.float32))
            for n in range(300)
        },
    }
    with db.connect() as conn:
        index.store(
            conn,
            Assembled(
                mailboxes=("kim-d", "rowe-p"),
                raw_files=(),
                rejected={},
                documents=tuple(documents),
                boilerplate=frozenset(),
            ),
            lambda doc: [doc.body],
        )
        rows = conn.execute(
            "SELECT c.id, d.external_id FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.source = 'test'"
        ).fetchall()
        index.write_vectors(
            conn, MODEL, [i for i, _ in rows], np.stack([vectors[e] for _, e in rows])
        )


def _force_hnsw(conn: psycopg.Connection[TupleRow]) -> None:
    ## On a table this small the planner reads the reader's few rows and sorts them exactly, so
    ## the bug can't show. Sorting and scanning made unaffordable, it walks the HNSW index, as it
    ## did on the 17k-chunk store where kean-s got nothing back.
    conn.execute("SET enable_seqscan = off")
    conn.execute("SET enable_sort = off")


def _found(person: str, as_of: date) -> dict[str, bool]:
    with db.connect(person) as conn:
        _force_hnsw(conn)
        lists = {
            "keyword": keyword.search(conn, "zephyr quokka", as_of, 10),
            "vector": vector.search(
                conn,
                _unit(np.array([0.8, 0.6, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
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
def test_a_filtered_hnsw_scan_still_reaches_the_one_readable_vector() -> None:
    ## 300 nearer vectors kim-d can't read sit between the query and the secret.
    with db.connect("kim-d") as conn:
        _force_hnsw(conn)
        hits = vector.search(conn, QUERY, MODEL, date(2001, 9, 30), 5)
        plan = conn.execute(
            sql.SQL("EXPLAIN {}").format(vector.ranked(MODEL, DIMS)),
            {"vector": HalfVector(QUERY), "cutoff": cutoff(date(2001, 9, 30)), "depth": 5},
        ).fetchall()
    ## Without the HNSW index in the plan this test proves nothing: it would pass with the bug.
    assert any("embedding_idx" in line for (line,) in plan)
    assert [CANARY in hit.text for hit in hits] == [True]
