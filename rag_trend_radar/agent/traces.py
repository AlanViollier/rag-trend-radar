"""Where finished runs are kept: one JSON file each, what page 2 replays and the evals score."""

from datetime import UTC, datetime
from pathlib import Path

from rag_trend_radar.config import get_settings
from rag_trend_radar.schema import AgentRun


def save(run: AgentRun, name: str) -> Path:
    """Write `run` under data/reports/agent/ as `<name>@<time>.json` and return the path."""
    folder = get_settings().data_dir / "reports" / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}@{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(run.model_dump_json(indent=1))
    return path
