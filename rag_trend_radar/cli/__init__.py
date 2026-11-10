"""rag-trend-radar: thin commands over the modules, one file per area."""

import typer

# Importing an area registers its commands on `app`.
from rag_trend_radar.cli import agents, evals, pipeline, radar, search  # noqa: F401
from rag_trend_radar.cli.app import app
from rag_trend_radar.exceptions import AppError

__all__ = ["app", "main"]


def main() -> None:
    """Run the app; a failure we know about prints its message, not a traceback."""
    try:
        app()
    except AppError as error:
        typer.echo(str(error), err=True)
        raise SystemExit(1) from error
