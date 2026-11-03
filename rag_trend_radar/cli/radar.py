"""Commands that build and read the radar."""

from rich import print as show

from rag_trend_radar.cli.app import app
from rag_trend_radar.db import connect as db


@app.command()
def radar_reset() -> None:
    """Empty the radar's tables and leave the store alone: mail, chunks and vectors stay."""
    db.reset_radar()
    show("radar tables recreated, empty")
