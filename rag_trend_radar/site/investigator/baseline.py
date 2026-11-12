"""Page 2's "ask without RAG": a run's question asked to the same model with no mail and no tools.

The prompt is the agent's with the tools taken out: the same asker, day, facts about the store and
answer rules, so the comparison is fair. The first try (2026-11-12) sent the bare question and the
model read "the mail on September 11" as the postal service. Paid, cached by model, prompt and
question: asking again costs nothing.
"""

import time
from datetime import date
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from rag_trend_radar import claude
from rag_trend_radar.agent.prompt import EMPLOYEE, INVESTIGATOR, held, verdict
from rag_trend_radar.db import connect as db
from rag_trend_radar.schema import AgentRun
from rag_trend_radar.site.investigator.records import Baseline

SYSTEM = """You answer questions about Enron's email for {who}. Today is {as_of}.

{store}

You can't search or read the mail here: answer from what you know about it.

Answer in plain words, without headings: a short opening line with the answer, then the \
evidence as a few sentences or a short list, with dates, names and short quotes. If you don't \
know, say so plainly instead of guessing.

End with one last line, exactly "Found: yes", "Found: partly" or "Found: no"."""
MAX_TOKENS = 1500
# One answer of a few hundred words, at Sonnet's price; shown before a paid run.
CENTS_EACH = 1


def prompt(conn: psycopg.Connection[TupleRow], person: str, as_of: date) -> str:
    """The system prompt for `person` on `as_of`, on a connection set to that person."""
    mail = held(conn, as_of)
    span = f"{mail.emails:,} emails from {mail.first} to {mail.last}"
    if person == db.INVESTIGATOR:
        # The radar is a tool too: the investigator is told only about the mail.
        who = INVESTIGATOR.removesuffix(" and the radar")
        return SYSTEM.format(
            who=who, as_of=as_of, store=f"The store holds {span} across 150 mailboxes."
        )
    return SYSTEM.format(
        who=EMPLOYEE.format(person=person), as_of=as_of, store=f"Your mailbox holds {span}."
    )


def cached(run: AgentRun, system: str) -> Baseline | None:
    """The answer asked before for this run's question and prompt, or None."""
    path = _path(run, system)
    return Baseline.model_validate_json(path.read_text()) if path.exists() else None


def ask(run: AgentRun, system: str) -> Baseline:
    """Ask Claude (paid, unless asked before) and keep the answer on disk."""
    if (saved := cached(run, system)) is not None:
        return saved
    begun = time.perf_counter()
    response = claude.client().messages.create(
        model=run.model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": run.question}],
    )
    text, found = verdict("".join(b.text for b in response.content if b.type == "text"))
    usage = response.usage
    answer = Baseline.model_validate(
        {
            "text": text,
            "found": found,
            "seconds": round(time.perf_counter() - begun, 1),
            "cost": claude.cost_usd(run.model, usage.input_tokens, usage.output_tokens),
            "system": system,
        }
    )
    path = _path(run, system)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(answer.model_dump_json())
    return answer


def _path(run: AgentRun, system: str) -> Path:
    return claude.cache_path("no_rag", f"{run.model}\n{system}\n{run.question}")
