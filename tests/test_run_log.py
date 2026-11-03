from pathlib import Path

import pytest

from rag_trend_radar.run_log import RunLog, summarise
from rag_trend_radar.schema import RadarRun, RunStage


def test_a_step_that_repeats_keeps_every_sample_and_its_items() -> None:
    log = RunLog("test", {"seed": 1})
    for n in (3, 4):
        with log.stage("a day") as handled:
            handled.items = n
    samples = log.run().stages["a day"]
    assert [s.items for s in samples] == [3, 4]
    assert all(s.seconds >= 0 for s in samples)


def test_a_step_that_fails_is_still_recorded() -> None:
    log = RunLog("test", {})

    def broken() -> None:
        with log.stage("broken") as handled:
            handled.items = 2
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        broken()
    assert log.run().stages["broken"][0].items == 2


def test_averages_come_from_the_samples() -> None:
    summary = summarise([RunStage(seconds=1.0, items=10), RunStage(seconds=3.0, items=10)])
    assert (summary.runs, summary.total_seconds, summary.mean_seconds) == (2, 4.0, 2.0)
    assert (summary.median_seconds, summary.max_seconds) == (2.0, 3.0)
    assert summary.ms_per_item == pytest.approx(200.0)


def test_a_step_that_counts_nothing_has_no_per_item_time() -> None:
    assert summarise([RunStage(seconds=1.0, items=0)]).ms_per_item == 0.0


def test_a_saved_run_reads_back_whole(tmp_path: Path) -> None:
    log = RunLog("test", {"min_size": 15, "selection": "eom"})
    log.results["topics"] = 7
    with log.stage("step") as handled:
        handled.items = 1
    saved = RadarRun.model_validate_json(log.save(tmp_path / "radar").read_text())
    assert saved.settings == {"min_size": 15, "selection": "eom"}
    assert saved.results == {"topics": 7}
    assert saved.peak_memory_mb > 0
