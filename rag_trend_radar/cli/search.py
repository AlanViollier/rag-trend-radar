"""The commands a person asks through: one search and an answer, or the agent."""

import json
from datetime import date

from rich import print as show
from rich.console import Console
from rich.markup import escape

from rag_trend_radar import agent as agents
from rag_trend_radar.answer import answer
from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import last_day
from rag_trend_radar.db import connect as db
from rag_trend_radar.embed import embed
from rag_trend_radar.schema import AgentStep
from rag_trend_radar.search import hybrid, pipeline


@app.command()
def ask(question: str, person: str = "investigator", as_of: str = "") -> None:
    """Answer a question as `person` on day `as_of` (default: the last finished day).

    `person` is 'investigator' or an employee's mailbox (kean-s): row-level security decides
    what the search can see, before Claude sees anything.
    """
    settings = get_settings()
    model, dims = settings.embedding_model, settings.embedding_dims
    query_vector = embed([question], model, dims, query=True)[0]
    with db.connect(person) as conn:
        day = date.fromisoformat(as_of) if as_of else last_day(conn)
        ranked = pipeline.reranked(
            conn, question, query_vector, model, day, settings.rerank_model, hybrid.DEPTH
        )
        hits = pipeline.final(ranked, hybrid.DEPTH)
        if not hits:
            show(f"Nothing in {person}'s mail up to {day} matches.")
            return
        result = answer(question, hits, day, settings.answer_model)
        titles: dict[int, str] = dict(
            conn.execute(
                "SELECT id, title FROM documents WHERE id = ANY(%s)",
                ([c.document_id for c in result.citations],),
            ).fetchall()
        )
    show(f"[bold]{person}[/bold], as of {day}\n\n{result.text}\n")
    for n, document_id in enumerate(dict.fromkeys(c.document_id for c in result.citations), 1):
        show(f"[{n}] {titles[document_id]} (document {document_id})")
    show(f"\n${result.cost_usd:.4f}")


# Steps go to stderr as they happen; the answer goes to stdout, so it pipes.
STEPS = Console(stderr=True, highlight=False)
SHOWN_THINKING = 400


@app.command()
def agent(question: str, person: str = "investigator", as_of: str = "", name: str = "ask") -> None:
    """Let the agent work a question as `person` on day `as_of`, showing each step as it goes.

    The run is saved under data/reports/agent/ as `<name>@<time>.json`. Asked again unchanged, it
    is read back from disk and costs nothing. The question sets' runs go through `agent-evals`.
    """
    with db.connect(person) as conn:
        day = date.fromisoformat(as_of) if as_of else last_day(conn)
        STEPS.print(f"[bold]{person}[/bold], as of {day}: {escape(question)}\n")
        run = agents.run(
            agents.Desk(conn, person, day), question, get_settings().answer_model, watch=_step
        )
        cited = list(dict.fromkeys(c.document_id for c in run.citations))
        titles: dict[int, str] = dict(
            conn.execute("SELECT id, title FROM documents WHERE id = ANY(%s)", (cited,)).fetchall()
        )
    path = agents.save(run, name)
    show(f"\n{escape(run.answer)}\n")
    for n, document_id in enumerate(cited, 1):
        show(f"[{n}] {escape(titles[document_id])} (document {document_id})")
    tools = sum(step.kind == "tool" for step in run.steps)
    STEPS.print(
        f"\nfound: {run.found} · {tools} tool calls · {run.seconds:.0f} s · ${run.cost_usd:.4f}"
        f"\nsaved {path}"
    )


def _step(step: AgentStep) -> None:
    if step.kind in ("thinking", "note"):
        thought = " ".join(step.text.split())
        cut = thought[:SHOWN_THINKING] + ("…" if len(thought) > SHOWN_THINKING else "")
        STEPS.print(f"[dim italic]{escape(cut)}[/dim italic]")
    elif step.kind == "tool":
        arguments = ", ".join(f"{k}={v!r}" for k, v in json.loads(step.arguments).items())
        mark = "[red]refused[/red]" if step.error else f"{step.seconds:.1f} s"
        STEPS.print(
            f"[#C2410C]→ {step.tool}[/#C2410C]({escape(arguments)})  {escape(step.summary)}  · {mark}"
        )
    else:
        cached = f", {step.cache_read:,} from cache" if step.cache_read else ""
        STEPS.print(
            f"[dim]claude: {step.input_tokens + step.cache_read + step.cache_write:,} in{cached},"
            f" {step.output_tokens:,} out · ${step.cost_usd:.4f} · {step.seconds:.1f} s[/dim]"
        )
