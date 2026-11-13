"""Where finished runs are kept: a question set's in the repo, any other run under data/."""

from datetime import UTC, datetime
from pathlib import Path

from rag_trend_radar.config import get_settings
from rag_trend_radar.records import AgentRun

# The latest run of every question in the sets, one file each, tracked: page 2 and the scores are
# built from them, and making them again costs real money.
KEPT = Path("curated/agent_runs")


def keep(run: AgentRun, question_id: str) -> Path:
    """Keep `run` as question `question_id`'s run, replacing the one before; return the path."""
    path = KEPT / f"{question_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=1) + "\n")
    return path


def kept(question_id: str) -> AgentRun | None:
    """Question `question_id`'s kept run, or None when it was never run."""
    path = KEPT / f"{question_id}.json"
    return AgentRun.model_validate_json(path.read_text()) if path.exists() else None


def save(run: AgentRun, name: str) -> Path:
    """Write a run asked outside the sets under data/reports/agent/ as `<name>@<time>.json`."""
    folder = get_settings().data_dir / "reports" / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}@{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(run.model_dump_json(indent=1))
    return path
