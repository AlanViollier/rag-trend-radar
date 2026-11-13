"""Commands that build page 2, the recorded investigator."""

import sys
from pathlib import Path

import typer
from rich import print as show

from rag_trend_radar import agent
from rag_trend_radar.cli.app import app
from rag_trend_radar.db import connect as db
from rag_trend_radar.pages import investigator, publish

PAGE = Path("docs/investigator/index.html")


@app.command()
def investigator_baseline(yes: bool = False) -> None:
    """Ask each picked run's question without RAG: same model and prompt, no mail, no tools.

    Paid, about a cent each; an answer kept with today's prompt is reused and costs nothing.
    """
    picked = [(pick.id, run) for pick in investigator.picks() if (run := agent.kept(pick.id))]
    todo = []
    for pick, run in picked:
        with db.connect(run.person) as conn:
            system = investigator.baseline.prompt(conn, run.person, run.as_of)
        if not investigator.baseline.kept(pick, system):
            todo.append((pick, run, system))
    cents = len(todo) * investigator.baseline.CENTS_EACH
    show(f"{len(todo)} of {len(picked)} not asked with today's prompt, about ${cents / 100:.2f}")
    if todo and not yes:
        typer.confirm("Ask them?", abort=True)
    for pick, run, system in todo:
        answer = investigator.baseline.ask(pick, run, system)
        show(f"{run.question[:60]}: {answer.found}, ${answer.cost:.4f}")


@app.command()
def investigator_export(out: Path = PAGE) -> None:
    """Write page 2 from the saved runs, their tool calls run again on the store (free, ~2 min)."""
    page = investigator.build(
        investigator.picks(), progress=lambda line: show(line, file=sys.stderr)
    )
    size = publish.write(investigator.TEMPLATE, page, out)
    runs = sum(len(g.runs) for g in page.groups)
    show(f"{out}: {runs} runs, {len(page.emails)} emails, {size / 1024:.0f} KB")
