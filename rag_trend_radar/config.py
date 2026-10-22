"""Single entry point for settings: config.yaml for shape, .env for secrets."""

from functools import cache
from pathlib import Path

import yaml
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_trend_radar.exceptions import ConfigError


class AppSettings(BaseSettings):
    """Typed settings. Built by `get_settings()`; tests construct it directly."""

    ## A misspelt key in config.yaml or .env fails at startup instead of being ignored.
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    data_dir: Path = Path("data")

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "radar"
    db_user: str = "radar"
    postgres_password: SecretStr = SecretStr("")

    timezone: str = "America/Chicago"

    embedding_model: str = "google/embeddinggemma-2"
    embedding_dims: int = 768
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    answer_model: str = "claude-sonnet-5-5"
    anthropic_api_key: SecretStr = SecretStr("")


@cache
def get_settings(config_path: Path = Path("config.yaml")) -> AppSettings:
    """Load settings from `config_path` plus the environment.

    Parameters
    ----------
    config_path:
        YAML file holding non-secret settings.

    Returns
    -------
    AppSettings
        Validated settings; raises `ConfigError` when the file doesn't validate.
    """
    values = yaml.safe_load(config_path.read_text()) or {}
    try:
        return AppSettings(**values)
    except ValidationError as error:
        raise ConfigError(f"{config_path}: {error}") from error
