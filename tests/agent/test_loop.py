"""The loop against a scripted Claude: no API, no database, every path a real run can take."""

from datetime import date
from pathlib import Path
from typing import Any, cast

import anthropic
import psycopg
import pytest
from anthropic.types.beta import BetaMessage

from rag_trend_radar import claude
from rag_trend_radar.agent import loop, prompt, tools
from rag_trend_radar.agent.desk import Desk, ToolResult, text
from rag_trend_radar.exceptions import AnswerRefused, ToolRefused

MODEL = "claude-sonnet-5-5"
DESK = Desk(cast(psycopg.Connection[Any], None), "investigator", date(2001, 12, 31))


def _message(stop: str, *blocks: dict[str, object]) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": stop,
            "stop_sequence": None,
            "content": list(blocks),
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def _thinking(words: str) -> dict[str, object]:
    return {"type": "thinking", "thinking": words, "signature": "sig"}


def _use(name: str, arguments: dict[str, object], use_id: str = "use-1") -> dict[str, object]:
    return {"type": "tool_use", "id": use_id, "name": name, "input": arguments}


def _answer(words: str, cites: int | None = None) -> dict[str, object]:
    citations = [
        {
            "type": "search_result_location",
            "source": f"document {cites}",
            "title": "Raptors",
            "cited_text": "the valuations",
            "search_result_index": 0,
            "start_block_index": 0,
            "end_block_index": 0,
        }
    ]
    return {"type": "text", "text": words, "citations": citations if cites else None}


class Scripted:
    """Stands in for anthropic.Anthropic: hands back its replies in order, keeps every request."""

    def __init__(self, *replies: BetaMessage) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def create(self, **request: Any) -> BetaMessage:
        # A copy: the loop goes on appending to the same list after the call.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.replies.pop(0)

    @property
    def client(self) -> anthropic.Anthropic:
        """Itself, typed as what the loop takes."""
        return cast(anthropic.Anthropic, self)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """No database, no real tools, runs cached under tmp_path; yields the tools called."""
    called: list[str] = []

    def fake_call(desk: Desk, name: str, arguments: object) -> ToolResult:
        called.append(name)
        if name == "nope":
            raise ToolRefused("no tool 'nope' for investigator")
        return ToolResult([text("12 emails match.")], "12 emails", (42,))

    monkeypatch.setattr(tools, "call", fake_call)
    monkeypatch.setattr(tools, "definitions", lambda desk: [])
    monkeypatch.setattr(loop, "system", lambda desk: "You answer questions.")
    monkeypatch.setattr(claude, "cache_path", lambda folder, key: tmp_path / f"{hash(key)}.json")
    return called


def test_a_tool_call_then_an_answer_leaves_a_whole_trace(offline: list[str]) -> None:
    client = Scripted(
        _message(
            "tool_use",
            _thinking("Count the Raptor mail."),
            _use("mail_counts", {"words": "raptor"}),
        ),
        _message("end_turn", _answer("Kaminski objected.", cites=42), _answer("\n\nFound: yes")),
    )
    seen: list[str] = []
    run = loop.run(
        DESK, "Who objected?", MODEL, client=client.client, watch=lambda s: seen.append(s.kind)
    )
    assert offline == ["mail_counts"]
    assert seen == ["thinking", "turn", "tool", "turn"]
    assert run.answer == "Kaminski objected."
    assert run.found == "yes"
    assert [c.document_id for c in run.citations] == [42]
    assert run.cost_usd == pytest.approx(2 * claude.cost_usd(MODEL, 1000, 100))
    result = client.requests[1]["messages"][-1]["content"][0]
    assert (result["tool_use_id"], result["content"][0]["text"]) == ("use-1", "12 emails match.")


def test_a_refused_tool_goes_back_as_an_error_and_the_run_goes_on() -> None:
    client = Scripted(
        _message("tool_use", _use("nope", {})),
        _message("end_turn", _answer("Nothing in the mail.\nFound: no")),
    )
    run = loop.run(DESK, "Anything?", MODEL, client=client.client)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert [s.error for s in run.steps if s.kind == "tool"] == [True]
    assert (run.found, run.citations) == ("no", ())


def test_out_of_tool_calls_the_next_request_allows_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "MAX_TOOL_CALLS", 1)
    client = Scripted(
        _message("tool_use", _use("search", {"query": "a"})),
        _message("end_turn", _answer("Partly.\nFound: partly")),
    )
    loop.run(DESK, "Q", MODEL, client=client.client)
    assert [r["tool_choice"]["type"] for r in client.requests] == ["auto", "none"]


def test_a_run_asked_again_is_read_back_without_calling_claude() -> None:
    first = loop.run(
        DESK, "Q", MODEL, client=Scripted(_message("end_turn", _answer("A\nFound: yes"))).client
    )
    seen: list[str] = []
    again = loop.run(
        DESK, "Q", MODEL, client=Scripted().client, watch=lambda s: seen.append(s.kind)
    )
    assert again == first
    assert seen == ["turn"]


def test_without_a_client_the_one_on_the_key_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Scripted(_message("end_turn", _answer("A\nFound: yes")))
    monkeypatch.setattr(claude, "client", lambda: client.client)
    loop.run(DESK, "Q", MODEL)
    assert len(client.requests) == 1


def test_a_declined_question_raises() -> None:
    with pytest.raises(AnswerRefused):
        loop.run(DESK, "Q", MODEL, client=Scripted(_message("refusal")).client)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Kaminski did.\n\nFound: yes", ("Kaminski did.", "yes")),
        ("Some of it.\nFound: Partly.", ("Some of it.", "partly")),
        ("No verdict here.", ("No verdict here.", None)),
        (
            "Found: no means nothing mid-text. Found: no",
            ("Found: no means nothing mid-text.", "no"),
        ),
    ],
)
def test_the_verdict_is_read_off_the_last_line(
    answer: str, expected: tuple[str, str | None]
) -> None:
    assert prompt.verdict(answer) == expected
