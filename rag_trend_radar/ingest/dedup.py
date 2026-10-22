"""One document per distinct email, however many mailboxes hold a copy."""

from rag_trend_radar.schema import ParsedEmail, SplitBody


def content_hash(email: ParsedEmail, body: SplitBody) -> str:
    """Fingerprint of sender, time, subject and own text: equal for every copy of one email."""
    raise NotImplementedError
