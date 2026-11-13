"""The agent's commands: one question worked step by step, a question set run, its runs scored."""

import json
from collections import defaultdict
from datetime import UTC, date, datetime

import typer
from rich import print as show
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_trend_radar import agent as agents
from rag_trend_radar.cli.app import app
from rag_trend_radar.config import get_settings
from rag_trend_radar.days import last_day
from rag_trend_radar.db import connect as db
from rag_trend_radar.evals import agent as checks
from rag_trend_radar.records import AgentQuestion, AgentScore, AgentStep

# The thorough runs of 2026-11-10 averaged about 15 cents a question; shown before a paid run.
CENTS_EACH = 15

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
        agents.keep(run, question.id)
        show(
            f"{n}/{len(asked)} {question.id}: {run.found}, ${run.cost_usd:.3f}, {run.seconds:.0f} s"
        )
    _report(name, asked)


@app.command()
def agent_scores(name: str = "scored") -> None:
    """Score the kept run of every question in set `name`; free, nothing is asked again."""
    _report(name, checks.questions(name))


def _report(name: str, asked: list[AgentQuestion]) -> None:
    scores: list[AgentScore] = []
    with db.connect() as conn:
        for question in asked:
            if (run := agents.kept(question.id)) is not None:
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
        f"{len(scores)} of {len(asked)} runs kept · all {checks.rate(scores)}"
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
