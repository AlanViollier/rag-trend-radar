"""Writing a page: its data put inside the HTML template, unless it holds a number it shouldn't."""

import re
from pathlib import Path

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.records import Record

# What published data must never hold. The mail is public, but 38 documents carry SSN-shaped
# numbers and 37 card-shaped ones; a page holding one isn't written.
SHAPES = {
    "SSN-shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card-shaped": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def write(template: Path, data: Record, out: Path) -> int:
    """Put the data into the page, unless it holds a number it shouldn't; return the page's bytes."""
    text = data.model_dump_json()
    for what, shape in SHAPES.items():
        if shape.search(text):
            raise ExportRefused(f"the page data holds a {what} number: nothing written")
    html = template.read_text().replace("__DATA__", text.replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return len(html.encode())
