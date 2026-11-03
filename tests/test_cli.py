from typer.main import get_command

from rag_trend_radar.cli import app


def test_every_command_is_registered() -> None:
    assert set(get_command(app).commands) == {  # type: ignore[attr-defined]
        "ask",
        "db-reset",
        "delete",
        "drop-vectors",
        "embed",
        "facts",
        "leaks",
        "questions",
        "radar-calibrate",
        "radar-cluster",
        "radar-replay",
        "radar-reset",
        "radar-runs",
        "recall",
        "replay",
        "scores",
        "stage",
        "vectors",
    }
