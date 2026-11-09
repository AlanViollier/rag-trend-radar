"""What every call to Claude shares: the client, what it costs, where its answer is cached."""

import hashlib
from pathlib import Path

import anthropic

from rag_trend_radar.config import get_settings

# US dollars per million tokens (Claude API, 2026-10-06), input then output.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}


def client() -> anthropic.Anthropic:
    """A client on the key in .env."""
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """What a call costs at the model's list price."""
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def cache_path(folder: str, key: str) -> Path:
    """Where the answer to a call goes, by everything it depends on (rule 6)."""
    digest = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
    return get_settings().data_dir / "cache" / folder / f"{digest}.json"
