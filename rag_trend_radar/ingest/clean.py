"""Split an email body into its own text and the replies it quotes."""

from rag_trend_radar.schema import SplitBody


def split_quoted(body: str) -> SplitBody:
    """Separate the new text from quoted replies and forwards (mail-parser-reply + Enron patterns)."""
    raise NotImplementedError
