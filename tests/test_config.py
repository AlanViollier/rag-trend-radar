from pathlib import Path

import pytest

from rag_trend_radar.config import get_settings
from rag_trend_radar.exceptions import ConfigError


def test_settings_load_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("data_dir: elsewhere\n")
    assert get_settings(config).data_dir == Path("elsewhere")


def test_unknown_key_fails_at_startup(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("data_dri: typo\n")
    with pytest.raises(ConfigError, match="data_dri"):
        get_settings(config)


def test_repo_config_validates() -> None:
    assert get_settings(Path("config.yaml")).embedding_model == "google/embeddinggemma-2"


def test_only_the_configured_model_loads_at_its_pinned_revision(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f"embedding_revision: {'a' * 40}\nrerank_revision: {'b' * 40}\n")
    settings = get_settings(config)
    assert settings.revision("google/embeddinggemma-2") == "a" * 40
    assert settings.revision("BAAI/bge-reranker-v2-m3") == "b" * 40
    assert settings.revision("sentence-transformers/all-MiniLM-L6-v2") == "main"


def test_the_cache_schema_cant_be_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("")
    monkeypatch.setenv("DB_SCHEMA", "cache")
    with pytest.raises(ConfigError, match="cache"):
        get_settings(config)
