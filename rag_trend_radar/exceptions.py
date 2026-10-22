"""Typed failures. Callers catch `AppError` when the policy is uniform, a subclass when it isn't."""


class AppError(Exception):
    """Base for every failure this package raises on purpose."""


class ConfigError(AppError):
    """Settings are missing or invalid."""
