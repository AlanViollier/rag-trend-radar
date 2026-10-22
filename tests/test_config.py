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
