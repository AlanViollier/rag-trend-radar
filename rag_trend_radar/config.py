"""Single entry point for settings: config.yaml for shape, .env for secrets."""

from functools import cache
from pathlib import Path

import yaml
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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
    ## Experiments get their own schema (exp_bare): the whole app runs against it, the store in
    ## public stays as it is. Interpolated into SQL, hence the pattern.
    db_schema: str = Field("public", pattern=r"^[a-z_][a-z0-9_]*$")
    postgres_password: SecretStr = SecretStr("")

    timezone: str = "America/Chicago"

    embedding_model: str = "google/embeddinggemma-2"
    embedding_dims: int = 768
    ## The model's commit on the Hugging Face Hub. Unpinned, a load fetches the latest, and an
    ## update between two runs would mix two models in one vectors table.
    embedding_revision: str = Field("main", pattern=r"^(?:main|[0-9a-f]{40})$")
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    ## Pinned for the same reason: cached rerank scores are keyed by it.
    rerank_revision: str = Field("main", pattern=r"^(?:main|[0-9a-f]{40})$")

    answer_model: str = "claude-sonnet-5-5"
    anthropic_api_key: SecretStr = SecretStr("")

    @field_validator("db_schema")
    @classmethod
    def _not_kept_schema(cls, schema: str) -> str:
        ## A reset drops the configured schema's tables: the cache must outlive every reset.
        if schema == "cache":
            raise ValueError("cache is kept across resets and can't be a store's schema")
        return schema

    def revision(self, model: str) -> str:
        """The revision to load `model` at: the pinned one for a configured model, else main."""
        pinned = {
            self.embedding_model: self.embedding_revision,
            self.rerank_model: self.rerank_revision,
        }
        return pinned.get(model, "main")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """The environment beats config.yaml (passed in as init values).

        Tests and CI point at another database with DB_NAME=radar_test, without editing the file.
        """
        return env_settings, dotenv_settings, init_settings, file_secret_settings


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
