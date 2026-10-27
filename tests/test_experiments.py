from collections.abc import Iterator

import pytest

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
