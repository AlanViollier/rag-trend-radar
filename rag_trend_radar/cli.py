"""rag-trend-radar: thin commands over the modules."""

from collections import Counter
from datetime import date
from functools import partial
from zoneinfo import ZoneInfo

import numpy as np
import typer
from rich import print as show
from rich.progress import track

from rag_trend_radar import index
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed, token_counter
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


@app.command("embed")
def embed_chunks(model: str = "", dims: int = 0, batch: int = 512) -> None:
    """Embed every chunk with no vector yet (default: the configured model and size)."""
    settings = get_settings()
    model, dims = model or settings.embedding_model, dims or settings.embedding_dims
    with db.connect() as conn:
        todo = list(index.unembedded(conn, model, dims).items())
        for start in track(range(0, len(todo), batch), description=f"{model} @ {dims}"):
            texts, ids = zip(*todo[start : start + batch], strict=True)
            vectors = embed(texts, model, dims)
            index.write_vectors(
                conn,
                model,
                [i for group in ids for i in group],
                np.repeat(vectors, [len(group) for group in ids], axis=0),
            )
            conn.commit()
    show(f"{sum(len(ids) for _, ids in todo):,} chunks embedded ({len(todo):,} distinct texts)")


@app.command()
def ask(question: str, person: str = "investigator", as_of: str = "") -> None:
    """Answer a question as `person` on day `as_of` (default: the last finished day)."""
    raise NotImplementedError
