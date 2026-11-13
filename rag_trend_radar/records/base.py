"""The base every record shares."""

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    """Frozen base: records are values, never mutated after construction."""

    model_config = ConfigDict(frozen=True)
