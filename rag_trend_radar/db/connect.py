"""Connections to the store, as the pipeline or as a person who is searching."""

import psycopg
from psycopg.rows import TupleRow


def connect(person: str | None = None) -> psycopg.Connection[TupleRow]:
    """Open a connection; with `person`, every query sees only what that person may see.

    Parameters
    ----------
    person:
        None for the pipeline (owner, no row filtering). 'investigator' or an employee's
        address for searches: the connection switches to radar_reader and sets app.person.
    """
    raise NotImplementedError


def reset() -> None:
    """Drop everything and recreate the tables from schema.sql."""
    raise NotImplementedError
