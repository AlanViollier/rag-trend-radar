"""Typed failures. Callers catch `AppError` when the policy is uniform, a subclass when it isn't."""


class AppError(Exception):
    """Base for every failure this package raises on purpose."""


class ConfigError(AppError):
    """Settings are missing or invalid."""


class ParseError(AppError):
    """A dataset file can't become a record: `reason` is the quarantine category, `detail` the
    value at fault (the date it gave), so quarantine groups by reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class AnswerRefused(AppError):
    """Claude declined to answer, fallbacks included; the message carries the stop details."""


class NoDaysLoaded(AppError):
    """A search needs a default day and the pipeline hasn't finished one."""


class EmbeddingFailed(AppError):
    """The model returned vectors that fail the sentinel check; nothing from that batch is kept."""
