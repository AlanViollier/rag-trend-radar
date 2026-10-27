from collections.abc import Iterator

import numpy as np
import pytest

from rag_trend_radar import index
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db


@pytest.fixture
def experiment(store: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with db.connect() as conn:
        conn.execute("INSERT INTO employees VALUES ('in-public') ON CONFLICT DO NOTHING")
    monkeypatch.setenv("DB_SCHEMA", "exp_isolation")
    get_settings.cache_clear()
    db.reset()
    yield
    monkeypatch.setenv("DB_SCHEMA", "public")
    get_settings.cache_clear()


@pytest.mark.usefixtures("experiment")
def test_an_experiment_gets_fresh_tables_and_leaves_the_store_alone() -> None:
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) FROM employees").fetchall() == [(0,)]
        conn.execute("INSERT INTO employees VALUES ('in-experiment')")
        assert conn.execute(
            "SELECT mailbox FROM public.employees WHERE mailbox = 'in-public'"
        ).fetchall() == [("in-public",)]


@pytest.mark.usefixtures("experiment")
def test_readers_can_search_inside_an_experiment() -> None:
    with db.connect("investigator") as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchall() == [(0,)]


@pytest.mark.usefixtures("experiment")
def test_vectors_go_to_the_experiment_even_when_the_store_has_the_table() -> None:
    with db.connect() as conn:
        ## The store has one already: an unqualified name would resolve to it and skip creation.
        conn.execute("CREATE TABLE IF NOT EXISTS public.vectors_fake_8 (chunk_id bigint)")
        index.write_vectors(conn, "test/fake", [], np.zeros((0, 8), dtype=np.float32))
        tables = conn.execute(
            "SELECT schemaname FROM pg_tables WHERE tablename = 'vectors_fake_8' ORDER BY 1"
        ).fetchall()
    assert ("exp_isolation",) in tables
