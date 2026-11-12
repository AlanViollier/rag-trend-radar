import pytest
from typer.main import get_command

from rag_trend_radar import cli
from rag_trend_radar.cli import app
from rag_trend_radar.exceptions import ConfigError


def test_every_command_is_registered() -> None:
    assert set(get_command(app).commands) == {  # type: ignore[attr-defined]
        "agent",
        "agent-evals",
        "agent-scores",
        "ask",
        "db-reset",
        "delete",
        "drop-vectors",
        "embed",
        "facts",
        "investigator-baseline",
        "investigator-export",
        "leaks",
        "questions",
        "radar-build",
        "radar-export",
        "radar-frames",
        "radar-reset",
        "radar-runs",
        "recall",
        "replay",
        "scores",
        "stage",
        "vectors",
    }


def test_a_known_failure_prints_its_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> None:
        raise ConfigError("no password set")

    monkeypatch.setattr(cli, "app", fail)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    assert "no password set" in capsys.readouterr().err
