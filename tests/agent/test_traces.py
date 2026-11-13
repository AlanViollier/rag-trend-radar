"""Kept runs: one file per question, replaced by the next run, read back whole."""

from pathlib import Path

import pytest

from rag_trend_radar.agent import traces
from tests.factories import agent_run


def test_a_kept_run_replaces_the_one_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traces, "KEPT", tmp_path)
    traces.keep(agent_run(answer="first"), "s-1")
    path = traces.keep(agent_run(answer="second"), "s-1")
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
