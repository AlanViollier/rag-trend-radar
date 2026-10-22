"""The Enron dataset (CMU, May 2015): one folder per mailbox, one file per email."""

from collections.abc import Iterator
from pathlib import Path

from rag_trend_radar.schema import ParsedEmail, RawFile


def iter_raw_files(root: Path) -> Iterator[RawFile]:
    """Yield every email file under `root` (the extracted maildir/), in path order."""
    raise NotImplementedError


def parse(raw: RawFile) -> ParsedEmail:
    """Read the headers and body of one file. Raises when the file can't be parsed."""
    raise NotImplementedError
