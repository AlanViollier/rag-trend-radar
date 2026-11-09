"""Database tests run against radar_test, never the store; CI makes them required."""

import os

import psycopg
import pytest
from psycopg import sql

# Before any settings load: the environment beats config.yaml, so every connection a test
# opens goes to radar_test. db.reset() wipes it, and the real store must never be in reach.
os.environ["DB_NAME"] = "radar_test"
os.environ["DB_SCHEMA"] = "public"

from rag_trend_radar.config import get_settings  # noqa: E402
from rag_trend_radar.db import connect as db  # noqa: E402


@pytest.fixture(scope="session")
def store() -> None:
    """A fresh radar_test with the schema; skipped when Postgres is down, unless CI requires it."""
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.db_name.endswith("_test"):
        pytest.fail(f"refusing to reset {settings.db_name}: tests only touch a *_test database")
    try:
        admin = psycopg.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname="postgres",
            user=settings.db_user,
            password=settings.postgres_password.get_secret_value(),
            autocommit=True,
            connect_timeout=3,
        )
    except psycopg.OperationalError as error:
        if os.environ.get("RADAR_REQUIRE_DB"):
            raise
        pytest.skip(f"Postgres not reachable, database tests skipped ({error})")
    with admin:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.db_name,)
        ).fetchall()
        if not exists:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.db_name)))
    db.reset()
