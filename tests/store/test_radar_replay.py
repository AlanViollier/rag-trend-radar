"""The radar's replay on a small invented world: two steady subjects, then a new one."""

from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.run_log import RunLog
from rag_trend_radar.trends import day
from rag_trend_radar.trends.moves import Centres

MODEL, DIMS = "test/replay", 8
CHICAGO = ZoneInfo("America/Chicago")
FIRST, SINCE, UNTIL = date(1990, 1, 1), date(1990, 2, 1), date(1990, 3, 25)
NEW_SUBJECT_FROM = date(1990, 3, 1)


def _vector(rng: np.random.Generator, axis: int) -> NDArray[np.float32]:
    v = np.eye(DIMS, dtype=np.float32)[axis] + rng.normal(0, 0.02, DIMS).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v), dtype=np.float32)


@pytest.fixture(scope="module")
def world(store: None) -> Iterator[None]:
    rng = np.random.default_rng(0)
    with db.connect() as conn:
        ids, vectors = [], []
        moment = FIRST
        n = 0
        while moment <= UNTIL:
            axes = [0, 1, 2] if moment >= NEW_SUBJECT_FROM else [0, 1]
            for axis in axes:
                for _ in range(4):
                    n += 1
                    sent = datetime(moment.year, moment.month, moment.day, 9, tzinfo=CHICAGO)
                    text = f"Subject: s{axis}\n\nnote {n}"
                    document = conn.execute(
                        "INSERT INTO documents (source, external_id, kind, sent_at, title, body)"
                        " VALUES ('replay-test', %s, 'email', %s, 'x', %s) RETURNING id",
                        (f"d{n}", sent, text),
                    ).fetchall()[0][0]
                    ids.append(
                        conn.execute(
                            "INSERT INTO chunks (document_id, position, text, text_hash)"
                            " VALUES (%s, 0, %s, %s) RETURNING id",
                            (document, text, f"replay-{n}"),
                        ).fetchall()[0][0]
                    )
                    vectors.append(_vector(rng, axis))
            moment += timedelta(days=1)
        index.write_vectors(conn, MODEL, ids, np.stack(vectors))
    yield
    db.reset_radar()
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'replay-test'")


def _setup(centres: Centres) -> day.Setup:
    return day.Setup(
        centres=centres, join=0.9, twin=0.97, parent=0.5, model=MODEL, dims=DIMS, min_size=10
    )


def _run(centres: Centres) -> RunLog:
    db.reset_radar()
    log = RunLog("test", {})
    with db.connect() as conn:
        day.replay(conn, _setup(centres), FIRST, UNTIL, SINCE, log)
    return log


def _rows(query: str) -> list[tuple[Any, ...]]:
    with db.connect() as conn:
        return list(conn.execute(query).fetchall())


@pytest.mark.usefixtures("world")
def test_steady_subjects_become_topics_and_a_new_one_is_born_when_a_week_closes() -> None:
    log = _run(Centres.FROZEN)
    assert log.weeks[0].topics == 2
    assert log.weeks[-1].topics == 3
    born = _rows("SELECT born_on FROM topics ORDER BY id")[-1][0]
    assert isinstance(born, date)
    assert born >= NEW_SUBJECT_FROM
    assert born.weekday() == day.WEEK_ENDS_ON
    assert _rows("SELECT count(*) FROM alerts WHERE kind = 'birth'") == [(1,)]


@pytest.mark.usefixtures("world")
def test_a_new_topic_keeps_the_mail_from_before_it_was_known_as_assigned_when_born() -> None:
    _run(Centres.FROZEN)
    early = _rows(
        "SELECT count(*) FROM assignments a JOIN topics t ON t.id = a.topic_id"
        " WHERE t.born_on >= '1990-03-01' AND a.sent_on < a.assigned_on"
    )
    assert early[0][0] > 0


@pytest.mark.usefixtures("world")
def test_a_frozen_centre_stays_and_a_recalculated_one_moves() -> None:
    query = "SELECT topic_id, count(DISTINCT centre::text) FROM topic_weeks GROUP BY 1 ORDER BY 1 LIMIT 1"
    _run(Centres.FROZEN)
    assert _rows(query)[0][1] == 1
    _run(Centres.LAST_28)
    assert _rows(query)[0][1] > 1


@pytest.mark.usefixtures("world")
def test_counts_divide_by_every_document_of_the_day() -> None:
    _run(Centres.FROZEN)
    assert _rows("SELECT documents FROM day_totals WHERE day = '1990-02-10'") == [(8,)]
    assert _rows("SELECT sum(documents) FROM topic_days WHERE day = '1990-02-10'") == [(8,)]


@pytest.mark.usefixtures("world")
@pytest.mark.parametrize("centres", [Centres.FROZEN, Centres.LAST_28])
def test_running_it_again_gives_the_same_radar(centres: Centres) -> None:
    query = (
        "SELECT t.born_on, a.chunk_id FROM assignments a JOIN topics t ON t.id = a.topic_id"
        " ORDER BY a.chunk_id"
    )
    _run(centres)
    first = _rows(query)
    _run(centres)
    assert _rows(query) == first


@pytest.mark.usefixtures("world")
def test_speed_and_heading_appear_once_a_topic_has_a_past() -> None:
    _run(Centres.FROZEN)
    assert _rows("SELECT count(*) FROM topic_weeks WHERE speed IS NOT NULL")[0][0] > 0
    assert _rows("SELECT count(*) FROM topic_weeks WHERE heading_id IS NOT NULL")[0][0] > 0


@pytest.mark.usefixtures("world")
def test_an_employee_sees_none_of_it() -> None:
    _run(Centres.FROZEN)
    with db.connect("kim-d") as conn:
        for table in ("topics", "assignments", "topic_days", "topic_weeks", "alerts"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchall() == [(0,)]  # noqa: S608
