"""rag-trend-radar: thin commands over the modules, one file per area."""

## Importing an area registers its commands on `app`.
from rag_trend_radar.cli import evals, pipeline, radar, search  # noqa: F401
from rag_trend_radar.cli.app import app

__all__ = ["app"]
