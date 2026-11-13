"""Writing a page: the data goes inside its template, can't break out of it, and is refused when
it holds a number shaped like private data."""

from datetime import date
from pathlib import Path

import pytest

from rag_trend_radar.exceptions import ExportRefused
from rag_trend_radar.pages import investigator, publish
from rag_trend_radar.pages.radar import compose
from rag_trend_radar.pages.radar.records import Event

TEMPLATES = [compose.TEMPLATE, investigator.TEMPLATE]


def _event(title: str) -> Event:
    return Event(id="x", day=date(2001, 1, 1), title=title)


@pytest.mark.parametrize("template", TEMPLATES)
def test_the_data_sits_inside_the_page(tmp_path: Path, template: Path) -> None:
    out = tmp_path / "docs" / "index.html"
    size = publish.write(template, _event("Something"), out)
    html = out.read_text()
    assert size == len(html.encode())
    assert "__DATA__" not in html
    assert _event("Something").model_dump_json() in html


@pytest.mark.parametrize("template", TEMPLATES)
def test_data_cant_close_the_script_it_sits_in(tmp_path: Path, template: Path) -> None:
    out = tmp_path / "index.html"
    publish.write(template, _event("a </script> b"), out)
    assert out.read_text().count("</script>") == template.read_text().count("</script>")


@pytest.mark.parametrize("number", ["123-45-6789", "4111 1111 1111 1111", "4111111111111111"])
def test_a_number_shaped_like_private_data_stops_the_write(tmp_path: Path, number: str) -> None:
    out = tmp_path / "index.html"
    with pytest.raises(ExportRefused, match="shaped"):
        publish.write(compose.TEMPLATE, _event(f"Account {number}"), out)
    assert not out.exists()
