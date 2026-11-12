"""Commands that run the agent on a question set and score its saved runs."""

from collections import defaultdict
from datetime import UTC, datetime

import typer
from rich import print as show
from rich.table import Table

from rag_trend_radar import agent as agents
from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.db import connect as db
from rag_trend_radar.evals import agent as checks
from rag_trend_radar.schema import AgentQuestion, AgentScore

# The thorough runs of 2026-11-10 averaged about 15 cents a question; shown before a paid run.
CENTS_EACH = 15


@app.command()
def agent_evals(name: str = "scored", only: str = "", yes: bool = False) -> None:
    """Run the agent on every question of set `name` (scored or curiosity), then score it.

    Paid: every question not saved before calls Claude. `only` keeps the questions whose id holds
    it. A question asked before unchanged is read back from disk and costs nothing.
    """
    asked = [q for q in checks.questions(name) if only in q.id]
    estimate = len(asked) * CENTS_EACH / 100
    show(f"{len(asked)} questions, about ${estimate:.2f} if none was asked before")
    if not yes:
        typer.confirm("Run them?", abort=True)
    model = get_settings().answer_model
    for n, question in enumerate(asked, 1):
        with db.connect(question.person) as conn:
            desk = agents.Desk(conn, question.person, question.as_of)
            run = agents.run(desk, question.question, model, planted=checks.planted(question))
        agents.save(run, question.id)
        show(
            f"{n}/{len(asked)} {question.id}: {run.found}, ${run.cost_usd:.3f}, {run.seconds:.0f} s"
        )
    _report(name, asked)


@app.command()
def agent_scores(name: str = "scored") -> None:
    """Score the latest saved run of every question in set `name`; free, nothing is asked again."""
    _report(name, checks.questions(name))


def _report(name: str, asked: list[AgentQuestion]) -> None:
    scores: list[AgentScore] = []
    with db.connect() as conn:
        for question in asked:
            if (run := agents.latest(question.id)) is not None:
                scores.append(checks.score(conn, question, run))
    table = Table("question", "kind", "passed", "found", "calls", "cents", "s", "note")
    for s in scores:
        flags = ("" if s.grounded else " UNGROUNDED") + (" HOLLOW" if s.hollow else "")
        table.add_row(
            s.id, s.kind, f"{s.passed}{flags}", str(s.found), str(s.tool_calls),
            f"{s.cost_usd * 100:.1f}", f"{s.seconds:.0f}", s.note[:60],
        )  # fmt: skip
    show(table)
    by_kind: dict[str, list[AgentScore]] = defaultdict(list)
    for s in scores:
        by_kind[s.kind].append(s)
    show(" · ".join(f"{kind} {checks.rate(group)}" for kind, group in by_kind.items()))
    cost = sum(s.cost_usd for s in scores)
    seconds = sorted(s.seconds for s in scores)
    show(
        f"{len(scores)} of {len(asked)} runs saved · all {checks.rate(scores)}"
        f" · ungrounded {sum(not s.grounded for s in scores)} · hollow {sum(s.hollow for s in scores)}"
        f" · ${cost:.2f} · median {seconds[len(seconds) // 2] if seconds else 0:.0f} s"
    )
    out = (
        get_settings().data_dir
        / "reports"
        / "agent_scores"
        / f"{name}@{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("[" + ",\n".join(s.model_dump_json() for s in scores) + "]\n")
    show(f"saved {out}")
