"""Kept runs: one file per question, replaced by the next run, read back whole."""

from datetime import date
from pathlib import Path

import pytest

from rag_trend_radar.agent import traces
from rag_trend_radar.schema import AgentRun


def _run(answer: str) -> AgentRun:
    return AgentRun(
        question="q", person="investigator", as_of=date(2002, 12, 31), model="m",
        prompt_version=9, steps=(), answer=answer, found="yes", citations=(), cost_usd=0.1,
        seconds=10.0, started="2026-11-10T00:00:00Z", commit="c",
    )  # fmt: skip


def test_a_kept_run_replaces_the_one_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traces, "KEPT", tmp_path)
    traces.keep(_run("first"), "s-1")
    path = traces.keep(_run("second"), "s-1")
    assert path == tmp_path / "s-1.json"
    assert path.read_text().endswith("\n")
    kept = traces.kept("s-1")
    assert kept is not None
    assert kept.answer == "second"


def test_a_question_never_run_has_no_kept_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traces, "KEPT", tmp_path)
    assert traces.kept("never") is None
