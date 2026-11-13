from collections.abc import Iterator
from datetime import date

import numpy as np
import pytest

from rag_trend_radar import index
from rag_trend_radar.db import connect as db
from rag_trend_radar.radar import clustering

MODEL, DIMS = "test/pool", 4
# Chicago: 23:30 on the last day of March is already April in UTC, and still counts as March.
ROWS = [
    ("kept", "2000-03-15 12:00-06", "Subject: Gas\n\nPrices rose.", 0),
    ("late-evening", "2000-03-31 23:30-06", "Subject: Gas\n\nPrices fell.", 0),
    ("short-reply", "2000-03-15 12:00-06", "Subject: Re\n\nok\n\nDid prices rise?", 15),
    ("bulk", "2000-03-15 12:00-06", "Subject: Hi\nFrom: noreply@x.commissioner.com\n\nScores.", 0),
    ("before", "2000-02-28 12:00-06", "Subject: Old\n\nToo early.", 0),
    ("after", "2000-04-01 00:30-06", "Subject: New\n\nToo late.", 0),
]


@pytest.fixture
def filled(store: None) -> Iterator[None]:
    with db.connect() as conn:
        ids = []
        for name, sent, text, quote in ROWS:
            document = conn.execute(
                "INSERT INTO documents (source, external_id, kind, sent_at, title, body)"
                " VALUES ('pool-test', %s, 'email', %s, %s, %s) RETURNING id",
                (name, sent, name, text),
            ).fetchall()[0][0]
            ids.append(
                conn.execute(
                    "INSERT INTO chunks (document_id, position, text, text_hash, quote_chars)"
                    " VALUES (%s, 0, %s, %s, %s) RETURNING id",
                    (document, text, f"pool-{name}", quote),
                ).fetchall()[0][0]
            )
        index.write_vectors(conn, MODEL, ids, np.eye(DIMS, dtype=np.float32)[[0] * len(ids)])
    yield
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = 'pool-test'")


@pytest.mark.usefixtures("filled")
def test_the_pool_holds_a_periods_chunks_that_say_something() -> None:
    with db.connect() as conn:
        pool = clustering.pool(conn, MODEL, DIMS, date(2000, 3, 1), date(2000, 3, 31))
        store = conn.execute("SELECT id::text FROM store").fetchall()
    assert [(pool.store,)] == store
    assert sorted(pool.texts) == ["Subject: Gas\n\nPrices fell.", "Subject: Gas\n\nPrices rose."]
    assert pool.vectors.shape == (2, DIMS)
