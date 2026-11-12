"""Commands that build page 2, the recorded investigator."""

import sys
from pathlib import Path

import typer
from rich import print as show

from rag_trend_radar import agent
from rag_trend_radar.cli.app import app
from rag_trend_radar.db import connect as db
from rag_trend_radar.site import investigator

RUNS = Path("evals/investigator_runs.jsonl")
PAGE = Path("docs/investigator/index.html")


@app.command()
def investigator_baseline(yes: bool = False) -> None:
    """Ask each picked run's question without RAG: same model and prompt, no mail, no tools.

    Paid, about a cent each; a question asked before is read back from disk and costs nothing.
    """
    runs = [run for pick in investigator.chosen(RUNS) if (run := agent.latest(pick.id))]
    prompts = []
    for run in runs:
        with db.connect(run.person) as conn:
            prompts.append(investigator.baseline.prompt(conn, run.person, run.as_of))
    todo = [
        (r, p) for r, p in zip(runs, prompts, strict=True) if not investigator.baseline.cached(r, p)
    ]
    cents = len(todo) * investigator.baseline.CENTS_EACH
    show(f"{len(todo)} of {len(runs)} not asked yet, about ${cents / 100:.2f}")
    if todo and not yes:
        typer.confirm("Ask them?", abort=True)
    for run, system in todo:
        answer = investigator.baseline.ask(run, system)
        show(f"{run.question[:60]}: {answer.found}, ${answer.cost:.4f}")


@app.command()
def investigator_export(out: Path = PAGE) -> None:
    """Write page 2 from the saved runs, their tool calls run again on the store (free, ~2 min)."""
    page = investigator.build(
        investigator.chosen(RUNS), progress=lambda line: show(line, file=sys.stderr)
    )
    size = investigator.write(page, out)
    runs = sum(len(g.runs) for g in page.groups)
    show(f"{out}: {runs} runs, {len(page.emails)} emails, {size / 1024:.0f} KB")
