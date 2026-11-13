"""Page 2, the recorded investigator: the chosen runs replayed, one file with its data inside."""

from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar import agent
from rag_trend_radar.days import local_day
from rag_trend_radar.db import connect as db
from rag_trend_radar.evals import agent as checks
from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.pages.investigator import baseline, replay
from rag_trend_radar.pages.investigator.records import (
    Baseline,
    ChosenRun,
    InvestigatorPage,
    PageEmail,
    PageGroup,
    PageRun,
    Step,
)
from rag_trend_radar.pages.radar.export import SHAPES
from rag_trend_radar.records import AgentRun

TEMPLATE = Path(__file__).with_name("page.html")
# The runs picked for the page, in page order.
PICKS = Path("curated/investigator/picks.jsonl")
_PLANTED_EMAIL = PageEmail(
    kind="planted",
    date=checks.PLANTED_DAY,
    subject=checks.PLANTED_SUBJECT,
    sender=checks.PLANTED_FROM,
)
_EMAILS = """
SELECT d.id, d.kind, d.sent_at, d.title, coalesce(e.sender, ''), coalesce(e.to_addrs, '{}'),
       coalesce(cardinality(e.cc_addrs), 0)
FROM documents d LEFT JOIN emails e ON e.document_id = d.id
WHERE d.id = ANY(%s)
"""


def chosen(path: Path) -> list[ChosenRun]:
    """The runs picked for the page, in page order."""
    return [ChosenRun.model_validate_json(line) for line in path.read_text().splitlines()]


def build(
    picked: Sequence[ChosenRun], progress: Callable[[str], None] = lambda _: None
) -> InvestigatorPage:
    """Every picked run replayed on the store as its asker, with its emails and its answer
    without RAG. Free: a pick with no kept run or no kept answer without RAG stops the export."""
    planted = {q.id for name in checks.SETS for q in checks.questions(name) if q.planted}
    groups: dict[str, list[PageRun]] = {}
    emails: dict[int, PageEmail] = {}
    for pick in picked:
        run = agent.kept(pick.id)
        if run is None:
            raise ExportRefused(f"{pick.id} has no kept run: run agent-evals first")
        with db.connect(run.person) as conn:
            asked = baseline.kept(pick.id, baseline.prompt(conn, run.person, run.as_of))
            if asked is None:
                raise ExportRefused(
                    f"{pick.id} has no answer without RAG for today's prompt: investigator-baseline"
                )
            steps = tuple(replay.steps(conn, run, staged=pick.id in planted))
            wanted = replay.named(steps) | {c.document_id for c in run.citations}
            emails |= _emails(conn, wanted - {checks.PLANTED})
        if checks.PLANTED in wanted:
            emails[checks.PLANTED] = _PLANTED_EMAIL
        page_run = _page_run(pick, run, steps, asked, staged=pick.id in planted)
        groups.setdefault(pick.group, []).append(page_run)
        progress(f"{pick.id}: {len(steps)} steps, {len(wanted)} emails")
    return InvestigatorPage(
        groups=tuple(PageGroup(title=t, runs=tuple(r)) for t, r in groups.items()),
        emails=emails,
    )


def write(page: InvestigatorPage, out: Path) -> int:
    """Put the data into the page, unless it holds a number it shouldn't; return the page's bytes."""
    text = page.model_dump_json()
    for what, shape in SHAPES.items():
        if shape.search(text):
            raise ExportRefused(f"the page data holds a {what} number: nothing written")
    html = TEMPLATE.read_text().replace("__DATA__", text.replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return len(html.encode())


def _page_run(
    pick: ChosenRun, run: AgentRun, steps: tuple[Step, ...], asked: Baseline, staged: bool
) -> PageRun:
    return PageRun(
        id=pick.id,
        label=pick.label,
        question=run.question,
        person=run.person,
        asker=pick.asker,
        staged=staged,
        steps=steps,
        answer=run.answer,
        found=run.found,
        citations=run.citations,
        cost_usd=round(run.cost_usd, replay.DIGITS + 1),
        seconds=round(run.seconds, 1),
        model=run.model,
        # The page shows neither the prompt nor more than four decimals of a cent.
        baseline=asked.model_copy(update={"cost": round(asked.cost, 5), "system": ""}),
    )


def _emails(conn: psycopg.Connection[TupleRow], wanted: set[int]) -> dict[int, PageEmail]:
    """The emails as the connection's person reads them; one they can't read stops the export."""
    rows = conn.execute(_EMAILS, (sorted(wanted),)).fetchall()
    if len(rows) != len(wanted):
        missing = sorted(wanted - {row[0] for row in rows})
        raise ExportRefused(f"emails the run named but its asker can't read: {missing}")
    return {
        doc: PageEmail(
            kind=kind, date=local_day(sent), subject=title, sender=sender, to=tuple(to), cc=cc
        )
        for doc, kind, sent, title, sender, to, cc in rows
    }
