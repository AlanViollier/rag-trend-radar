"""The base every record shares, and reading a file of them."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Frozen base: records are values, never mutated after construction."""

    model_config = ConfigDict(frozen=True)


def read_records[T: Record](path: Path, record: type[T]) -> list[T]:
    """A file of records, one JSON object a line, in file order."""
    return [record.model_validate_json(line) for line in path.read_text().splitlines()]
