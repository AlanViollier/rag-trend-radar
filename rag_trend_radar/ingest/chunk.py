"""Cut text into pieces the embedding model takes whole."""


def chunk(text: str, max_tokens: int) -> list[str]:
    """Return `text` as one piece when it fits, otherwise split it with semchunk."""
    raise NotImplementedError
