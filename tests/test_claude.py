import pytest

from rag_trend_radar import claude
from rag_trend_radar.config import get_settings


def test_a_call_costs_list_price() -> None:
    assert claude.cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)


def test_cache_names_stay_as_they_were() -> None:
    # Answers already paid for are found by this name: changing the hash would pay for them again.
    path = claude.cache_path("answers", "key")
    assert path == get_settings().data_dir / "cache" / "answers" / (
        "82af27c64245bfd15a520134b44e6bb1.json"
    )


def test_cached_tokens_cost_their_own_prices() -> None:
    # 1M written to the cache at 1.25 x $2, 1M read back at $0.10.
    assert claude.cost_usd(
        "claude-sonnet-5-5", 0, 0, cache_write=1_000_000, cache_read=1_000_000
    ) == pytest.approx(2.6)
