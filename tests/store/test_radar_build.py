"""The cluster tables from frames made by hand: what's written, and that a rebuild changes nothing."""

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.radar import build, frames
from rag_trend_radar.radar.clustering import Pool
from rag_trend_radar.radar.frames import Frame
from tests.factories import take

MODEL, DIMS = "test/build", 8
START = date(2001, 1, 7)
WEEKS = [START + timedelta(weeks=i) for i in range(12)]
SIZE = 30


@pytest.fixture
def world(store: None) -> Iterator[tuple[Pool, list[int]]]:
    """A cluster about gas sent by many, a newsletter from one sender, with mail spread over the weeks."""
    rng = np.random.default_rng(0)
    axes = np.eye(DIMS, dtype=np.float32)
    ids, documents, texts, vectors, days = [], [], [], [], []
    with db.connect() as conn:
        conn.execute("INSERT INTO employees (mailbox) VALUES ('build-a') ON CONFLICT DO NOTHING")
        for n in range(2 * SIZE):
            gas = n < SIZE
            sender = f"s{n}@b.c" if gas else "letter@b.c"
            text = f"Subject: {'Gas' if gas else 'Weekly'} {n}\nFrom: {sender}\n\nPrices rose."
            sent = START + timedelta(days=n % 7)
            document = conn.execute(
                "INSERT INTO documents (source, external_id, kind, sent_at, title, body)"
                " VALUES ('build-test', %s, 'email', %s, 'x', %s) RETURNING id",
                (str(n), f"{sent} 12:00-06", text),
            ).fetchall()[0][0]
            conn.execute(
                "INSERT INTO emails (document_id, message_id, sender) VALUES (%s, %s, %s)",
                (document, str(n), sender),
            )
            conn.execute(
                "INSERT INTO access (document_id, mailbox, from_at) VALUES (%s, 'build-a', %s)",
                (document, sent),
            )
            chunk = conn.execute(
                "INSERT INTO chunks (document_id, position, text, text_hash)"
                " VALUES (%s, 0, %s, %s) RETURNING id",
                (document, text, f"build-{n}"),
            ).fetchall()[0][0]
            vector = axes[0 if gas else 1] + rng.normal(0, 0.01, DIMS)
            ids.append(chunk)
            documents.append(document)
            texts.append(text)
            vectors.append(vector / np.linalg.norm(vector))
            days.append(sent)
        index.write_vectors(conn, MODEL, ids, np.array(vectors, dtype=np.float32))
        made_in = str(conn.execute("SELECT id FROM store").fetchall()[0][0])
    n = len(ids)
    yield (
        Pool(
            np.array(ids, dtype=np.int64),
            np.array(documents, dtype=np.int64),
            texts,
            np.array(vectors, dtype=np.float32),
            days,
            [False] * n,
            made_in,
        ),
        ids,
    )
    db.reset_radar()
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'build-test'")


def _frames(weeks: list[date], agree: bool = True) -> list[Frame]:
    labels = np.array([0] * SIZE + [1 if agree else -1] * SIZE, dtype=np.int64)
    rows = np.arange(2 * SIZE, dtype=np.int64)
    return [Frame(week, rows, labels) for week in weeks]


def _dump(table: str) -> list[tuple[object, ...]]:
    with db.connect() as conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()  # noqa: S608


def test_a_build_writes_clusters_weeks_chunks_and_days(world: tuple[Pool, list[int]]) -> None:
    chosen, ids = world
    made = _frames(WEEKS)
    with db.connect() as conn:
        found = build.write(conn, chosen, made, made, MODEL)
    assert found["clusters"] == 2
    assert found["cluster_weeks"] == 2 * len(WEEKS)
    assert found["cluster_chunks"] == len(ids)
    with db.connect() as conn:
        kinds = conn.execute("SELECT kind, shown FROM clusters ORDER BY id").fetchall()
        held = conn.execute(
            "SELECT min(first_week), max(last_week) FROM cluster_chunks WHERE cluster_id = 0"
        ).fetchall()
        day = conn.execute(
            "SELECT sum(documents), sum(chunks), max(mailboxes), sum(radar_chunks) FROM day_totals"
        ).fetchall()
    assert kinds == [("topic", True), ("newsletter", False)]
    assert held == [(WEEKS[0], WEEKS[-1])]
    assert day == [(len(ids), len(ids), 1, len(ids))]


def test_a_rebuild_changes_nothing(world: tuple[Pool, list[int]]) -> None:
    chosen, _ = world
    made = _frames(WEEKS)
    tables = ("clusters", "cluster_weeks", "cluster_chunks", "day_totals", "alerts")
    with db.connect() as conn:
        build.write(conn, chosen, made, made, MODEL)
    first = {t: _dump(t) for t in tables}
    with db.connect() as conn:
        build.write(conn, chosen, made, made, MODEL)
    assert {t: _dump(t) for t in tables} == first


def test_a_cluster_the_other_seed_never_found_isnt_shown(world: tuple[Pool, list[int]]) -> None:
    chosen, _ = world
    with db.connect() as conn:
        build.write(conn, chosen, _frames(WEEKS), _frames(WEEKS, agree=False), MODEL)
    with db.connect() as conn:
        shown = conn.execute("SELECT id, shown FROM clusters ORDER BY id").fetchall()
    assert shown[0] == (0, True)
    assert shown[1] == (1, False)


def test_a_short_lived_cluster_isnt_shown_unless_the_mail_ended_first(
    world: tuple[Pool, list[int]],
) -> None:
    chosen, _ = world
    rows = np.arange(2 * SIZE, dtype=np.int64)
    steady = np.array([0] * SIZE + [-1] * SIZE, dtype=np.int64)
    both = np.array([0] * SIZE + [1] * SIZE, dtype=np.int64)
    middle = [Frame(w, rows, both if 4 <= i < 7 else steady) for i, w in enumerate(WEEKS)]
    with db.connect() as conn:
        build.write(conn, chosen, middle, middle, MODEL)
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) FROM clusters WHERE shown").fetchall() == [(1,)]
    final = [Frame(w, rows, both if i >= 10 else steady) for i, w in enumerate(WEEKS)]
    with db.connect() as conn:
        build.write(conn, chosen, final, final, MODEL)
    with db.connect() as conn:
        kinds = conn.execute("SELECT shown FROM clusters ORDER BY id").fetchall()
    assert kinds == [(True,), (False,)]


def test_a_delete_then_a_rebuild_takes_the_chunk_out_and_leaves_other_clusters_their_mail(
    world: tuple[Pool, list[int]], tmp_path: Path
) -> None:
    chosen, ids = world
    (tmp_path / frames.STAMP).write_text(chosen.store)
    for made in _frames(WEEKS):
        np.save(tmp_path / f"{made.end}-chunks.npy", chosen.chunk_ids[made.rows])
        np.save(tmp_path / f"{made.end}-labels-{frames.CLUSTER_VERSION}.npy", made.labels)
    saved = frames.load(chosen, WEEKS, tmp_path)
    with db.connect() as conn:
        build.write(conn, chosen, saved, saved, MODEL)
    letters = _without_surge(_dump("cluster_weeks")[len(WEEKS) :])
    gone = ids[3]
    with db.connect() as conn:
        document = conn.execute("SELECT document_id FROM chunks WHERE id = %s", (gone,)).fetchall()
        index.delete_document(conn, document[0][0])
    left = take(chosen, np.flatnonzero(chosen.chunk_ids != gone))
    after = frames.load(left, WEEKS, tmp_path)
    with db.connect() as conn:
        build.write(conn, left, after, after, MODEL)
        held = conn.execute("SELECT count(*) FROM cluster_chunks WHERE chunk_id = %s", (gone,))
        sizes = conn.execute("SELECT DISTINCT chunks FROM cluster_weeks WHERE cluster_id = 0")
        assert (held.fetchall(), sizes.fetchall()) == ([(0,)], [(SIZE - 1,)])
    # Surge moves a little: it compares each cluster against all mail, now one chunk less.
    assert _without_surge(_dump("cluster_weeks")[len(WEEKS) :]) == letters


def _without_surge(rows: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    return [row[:9] + row[10:] for row in rows]
