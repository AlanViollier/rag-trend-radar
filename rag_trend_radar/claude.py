"""What every call to Claude shares: the client, what it costs, where its answer is cached."""

import hashlib
from pathlib import Path

import anthropic

from rag_trend_radar.config import get_settings

# US dollars per million tokens (Claude API, 2026-11-03): input, output, cache read.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.10), "claude-opus-5-5": (4.0, 20.0, 0.20)}
# Writing the 5-minute prompt cache costs this much more than plain input.
CACHE_WRITE = 1.25


def client() -> anthropic.Anthropic:
    """A client on the key in .env."""
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
) -> float:
    """What a call costs at the model's list price; `input_tokens` excludes cached ones."""
    price_in, price_out, price_read = PRICES[model]
    spent = (
        input_tokens * price_in
        + output_tokens * price_out
        + cache_write * price_in * CACHE_WRITE
        + cache_read * price_read
    )
    return spent / 1e6


def cache_path(folder: str, key: str) -> Path:
    """Where the answer to a call goes, by everything it depends on: no call is paid for twice."""
    digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
    return get_settings().data_dir / "cache" / folder / f"{digest}.json"
