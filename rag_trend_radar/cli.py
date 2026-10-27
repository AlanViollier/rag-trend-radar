"""rag-trend-radar: thin commands over the modules."""

from collections import Counter
from datetime import date
from functools import partial
from zoneinfo import ZoneInfo

import typer
from rich import print as show

from rag_trend_radar import index
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import token_counter
from rag_trend_radar.ingest.assemble import assemble
from rag_trend_radar.ingest.chunk import pieces

app = typer.Typer(no_args_is_help=True)


@app.command()
def db_reset() -> None:
    """Drop everything and recreate the tables."""
    db.reset()


@app.command()
def load(since: str = "2001-09-01", until: str = "2001-09-30") -> None:
    """Read the Enron emails sent from `since` to `until` into the store (default: the M1 slice)."""
    settings = get_settings()
    assembled = assemble(
        settings.data_dir / "enron" / "maildir",
        date.fromisoformat(since),
        date.fromisoformat(until),
        ZoneInfo(settings.timezone),
    )
    chunker = partial(
        pieces,
        count_tokens=token_counter(settings.embedding_model),
        boilerplate=assembled.boilerplate,
    )
    with db.connect() as conn:
        chunks = index.store(conn, assembled, chunker)
    kinds = Counter(d.kind for d in assembled.documents)
    show(
        f"{len(assembled.raw_files):,} files ({len(assembled.rejected):,} quarantined) → "
        f"{kinds['email']:,} emails + {kinds['quoted']:,} quoted messages → {chunks:,} chunks"
    )


@app.command()
def ask(question: str, person: str = "investigator", as_of: str = "") -> None:
    """Answer a question as `person` on day `as_of` (default: the last finished day)."""
    raise NotImplementedError
