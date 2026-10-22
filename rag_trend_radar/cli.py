"""rag-trend-radar: thin commands over the modules."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def db_reset() -> None:
    """Drop everything and recreate the tables."""
    raise NotImplementedError


@app.command()
def load(limit: int = 0) -> None:
    """Read Enron files into raw_files, then documents and chunks (0 = everything)."""
    raise NotImplementedError


@app.command()
def ask(question: str, person: str = "investigator", as_of: str = "") -> None:
    """Answer a question as `person` on day `as_of` (default: the last finished day)."""
    raise NotImplementedError
