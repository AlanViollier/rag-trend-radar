"""The labelled questions retrieval is measured on."""

from datetime import date

from rag_trend_radar.records.base import Record


class EvalQuestion(Record):
    """A labelled question: asked by someone on a day, answered by known documents.

    Documents are named by external id (the content hash), which survives rebuilding the store;
    row ids don't.
    """

    id: str
    question: str
    as_of: date
    person: str
    relevant: tuple[str, ...]
    # The identifier a codes-style question quotes, as written; empty for the other styles.
    code: str = ""


class FactQuestion(Record):
    """A hard question: what its results must hold, must not hold, or that nothing should match."""

    id: str
    question: str
    as_of: date
    person: str = "investigator"
    facts: tuple[str, ...] = ()
    needs_mail: bool = False
    forbid: tuple[str, ...] = ()
    absent: bool = False


class FactResult(Record):
    """One hard question checked against one list of results."""

    id: str
    missing: tuple[str, ...]
    leaked: tuple[str, ...]
    top_score: float
    mail: int
