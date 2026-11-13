"""Page 2's export without a store: the picked runs, the replayed steps, the prompt, the write."""

from datetime import date
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest

from rag_trend_radar.agent import tools
from rag_trend_radar.agent.desk import ToolResult, text
from rag_trend_radar.answer import cited
from rag_trend_radar.evals import agent as checks
from rag_trend_radar.exceptions import ExportRefused, ToolRefused
from rag_trend_radar.pages import investigator
from rag_trend_radar.pages.investigator import baseline, replay, timings
from rag_trend_radar.pages.investigator.records import (
    Baseline,
    Hit,
    Note,
    Thought,
    ToolCall,
    Turn,
)
from rag_trend_radar.records import AgentStep, SearchStage
from tests.factories import agent_run

CONN = cast(psycopg.Connection[Any], None)


class _Held:
    """A connection whose one query is the count of mail a person can read."""

    def execute(self, *_: object) -> _Held:
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return [(1234, None, None)]


def _search(*documents: int) -> ToolResult:
    blocks = []
    for doc in documents:
        block = cited(doc, f"Subject: s{doc}\nFrom: a@enron.com\n\nbody {doc}")
        block["title"] = f"2001-10-26 · {'press' if doc % 2 else 'mail'} · {block['title']}"
        blocks.append(block)
    stages = (SearchStage(name="embed", seconds=0.123456789, items=1),)
    return ToolResult(blocks, "2 results", tuple(documents), stages=stages)


def _tool(name: str, documents: tuple[int, ...] = (), arguments: str = "{}") -> AgentStep:
    return AgentStep(
        kind="tool", tool=name, arguments=arguments, documents=documents, seconds=1.23456
    )


def test_the_picked_runs_are_questions_of_the_sets_once_each() -> None:
    picked = investigator.picks()
    ids = [p.id for p in picked]
    asked = {q.id for name in checks.SETS for q in checks.questions(name)}
    assert len(ids) == len(set(ids))
    assert set(ids) <= asked


def test_steps_keep_their_order_and_rounded_times(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "call", lambda *_: _search(7, 8))
    run = agent_run(
        AgentStep(kind="thinking", text=" plan \n"),
        AgentStep(kind="note", text="looking"),
        AgentStep(kind="turn", seconds=2.123456),
        _tool("search", (7, 8)),
    )
    thought, note, turn, call = replay.steps(CONN, run, staged=False)
    assert thought == Thought(text="plan")
    assert note == Note(text="looking")
    assert turn == Turn(seconds=2.123)
    assert isinstance(call, ToolCall)
    assert call.seconds == 1.235
    assert call.stages[0].seconds == 0.123
    assert call.hits == (Hit(document_id=7, origin="press"), Hit(document_id=8, origin="mail"))


def test_a_planted_email_shows_second_in_the_first_search_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "call", lambda *_: _search(7, 8))
    run = agent_run(_tool("search", (7, 8)), _tool("search", (7, 8)))
    first, second = replay.steps(CONN, run, staged=True)
    assert isinstance(first, ToolCall)
    assert isinstance(second, ToolCall)
    assert [h.document_id for h in first.hits] == [7, checks.PLANTED, 8]
    assert [h.document_id for h in second.hits] == [7, 8]
    assert replay.named([first, second]) == {7, 8, checks.PLANTED}


def test_other_emails_than_the_run_saw_stop_the_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "call", lambda *_: _search(7))
    with pytest.raises(ExportRefused, match="other emails"):
        replay.steps(CONN, agent_run(_tool("search", (9,))), staged=False)


def test_read_email_and_counts_keep_what_they_showed(monkeypatch: pytest.MonkeyPatch) -> None:
    results = {
        "read_email": ToolResult([text("Subject: x")], "read", (5,)),
        "mail_counts": ToolResult([text("3 emails match.\nMonth  emails\n2001-10  3")], "3"),
    }
    monkeypatch.setattr(tools, "call", lambda _desk, name, _args: results[name])
    read, counted = replay.steps(
        CONN, agent_run(_tool("read_email", (5,)), _tool("mail_counts")), staged=False
    )
    assert isinstance(read, ToolCall)
    assert read.read == 5
    assert isinstance(counted, ToolCall)
    assert counted.text.startswith("3 emails match.")


def test_a_call_refused_then_is_refused_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> ToolResult:
        raise ToolRefused("no tool 'alerts' for kean-s")

    monkeypatch.setattr(tools, "call", refuse)
    (call,) = replay.steps(CONN, agent_run(_tool("alerts")), staged=False)
    assert isinstance(call, ToolCall)
    assert call.text == "no tool 'alerts' for kean-s"


def test_the_prompt_without_rag_tells_of_the_mail_but_not_the_tools() -> None:
    held = cast(psycopg.Connection[Any], _Held())
    told = baseline.prompt(held, "investigator", date(2002, 12, 31))
    assert "1,234 emails" in told
    assert "150 mailboxes" in told
    assert "radar" not in told
    assert "search" not in told.split("You can't")[0]
    mine = baseline.prompt(held, "kean-s", date(2002, 12, 31))
    assert "kean-s, an Enron employee" in mine
    assert "Your mailbox holds 1,234 emails" in mine


def test_an_answer_kept_with_another_prompt_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline, "KEPT", tmp_path)
    saved = Baseline(text="no", found="no", seconds=1.0, cost=0.01, system="prompt v1")
    (tmp_path / "c-raptor.json").write_text(saved.model_dump_json())
    assert baseline.kept("c-raptor", "prompt v1") == saved
    assert baseline.kept("c-raptor", "prompt v2") is None
    assert baseline.kept("c-other", "prompt v1") is None


def test_kept_search_times_replace_the_ones_measured_now(monkeypatch: pytest.MonkeyPatch) -> None:
    read = ToolResult([text("Subject: x")], "read", (7,))
    monkeypatch.setattr(
        tools, "call", lambda _desk, name, _args: read if name == "read_email" else _search(7, 8)
    )
    run = agent_run(_tool("search", (7, 8)), _tool("read_email", (7,)), _tool("search", (7, 8)))
    measured = replay.steps(CONN, run, staged=False)
    assert (
        replay.search_times(measured).searches
        == ((SearchStage(name="embed", seconds=0.123, items=1),),) * 2
    )
    first = (SearchStage(name="embed", seconds=1.5, items=1),)
    second = (SearchStage(name="embed", seconds=2.5, items=1),)
    kept = replay.steps(
        CONN, run, staged=False, times=timings.SearchTimes(searches=(first, second))
    )
    assert replay.search_times(kept).searches == (first, second)


def test_search_times_are_kept_per_pick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timings, "KEPT", tmp_path)
    assert timings.kept("c-raptor") is None
    times = timings.SearchTimes(searches=((SearchStage(name="rerank", seconds=0.5, items=50),),))
    timings.keep("c-raptor", times)
    assert timings.kept("c-raptor") == times
