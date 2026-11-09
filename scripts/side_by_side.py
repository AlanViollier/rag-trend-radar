"""Write an HTML page of sampled emails, raw next to cleaned, for reading the cleaner's work.

Usage: uv run python scripts/side_by_side.py data/enron/maildir > data/reports/side_by_side.html
"""

import html
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.ingest.clean import (
    BOILERPLATE_MIN_EMAILS,
    frequent_lines,
    normalize,
    split_quoted,
    strip_boilerplate,
)
from rag_trend_radar.ingest.dedup import content_hash
from rag_trend_radar.schema import ParsedEmail, RawFile, SplitBody
from rag_trend_radar.sources import enron

RANDOM, WITH_QUOTES = 25, 25
RAW_CAP = 4000  # chars of the raw body shown; the rest is summarised


def main(root: Path) -> None:
    """Sample emails from `root`, clean them, print the page."""
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    unique: dict[str, tuple[str, str, int]] = {}
    with ProcessPoolExecutor() as pool:
        for result in pool.map(_own, [(root, p) for p in paths], chunksize=2000):
            if result is not None:
                key, rel, own, blocks = result
                unique.setdefault(key, (rel, own, blocks))
    owns = [own for _rel, own, _blocks in unique.values()]
    frequent = frozenset(frequent_lines(((date.min, own) for own in owns), BOILERPLATE_MIN_EMAILS))
    known = {_collapse(own) for own in owns if own}

    rng = random.Random(0)  # noqa: S311 (a reproducible sample, not a secret)
    rows = list(unique.values())
    picked_random = rng.sample(rows, RANDOM)
    picked_quotes = rng.sample([r for r in rows if r[2] and r not in picked_random], WITH_QUOTES)

    sections = [
        ("01", "Random emails", picked_random),
        ("02", "Emails that quote or forward", picked_quotes),
    ]
    body = "".join(
        f'<h2><span class="n">{n} ·</span> {label}</h2><hr>'
        + "".join(_card(root, rel, frequent, known) for rel, _own, _blocks in picked)
        for n, label, picked in sections
    )
    meta = [
        ("distinct emails", f"{len(unique):,}"),
        ("boilerplate lines", f"{len(frequent):,} (in ≥ {BOILERPLATE_MIN_EMAILS} emails)"),
        ("sample", f"{RANDOM} random + {WITH_QUOTES} with quoted messages, seed 0"),
    ]
    print(_page(meta, body))


# Worker: one file to its dedup key and own text.


def _own(job: tuple[Path, Path]) -> tuple[str, str, str, int] | None:
    root, path = job
    rel = path.relative_to(root).as_posix()
    try:
        parsed = _parse(root, rel)
    except ParseError:
        return None
    split = _split(parsed)
    return content_hash(parsed.subject, parsed.body), rel, split.own, len(split.quoted)


def _parse(root: Path, rel: str) -> ParsedEmail:
    mailbox, *folders, _name = rel.split("/")
    raw = RawFile(
        path=rel, mailbox=mailbox, folder="/".join(folders), content=(root / rel).read_bytes()
    )
    return enron.parse(raw)


def _split(parsed: ParsedEmail) -> SplitBody:
    return split_quoted(normalize(parsed.body))


def _collapse(text: str) -> str:
    return " ".join(text.split())


# HTML.


def _card(root: Path, rel: str, frequent: frozenset[str], known: set[str]) -> str:
    parsed = _parse(root, rel)
    split = _split(parsed)
    kept = strip_boilerplate(split.own, frequent)
    trimmed = split.own[len(kept) :].strip("\n")
    raw = parsed.body.strip("\n")
    more = f'<p class="meta">… {len(raw) - RAW_CAP:,} more chars</p>' if len(raw) > RAW_CAP else ""

    cleaned = f'<pre class="own">{_e(kept) or "<span class=meta>(no own text)</span>"}</pre>'
    if trimmed:
        cleaned += (
            f'<p class="tag">→ trimmed as boilerplate</p><pre class="cut">{_e(trimmed)}</pre>'
        )
    for block in split.quoted:
        orphan = _collapse(block.text) not in known
        tag = "orphan → chunked (rule A)" if orphan else "already an email → not chunked"
        cleaned += (
            f'<p class="tag{" hot" if orphan else ""}">→ quoted · {tag}</p>'
            f'<pre class="head">{_e(block.header)}</pre><pre class="quoted">{_e(block.text)}</pre>'
        )
    return (
        f'<section class="card"><div class="row"><span class="path">{_e(rel)}</span>'
        f'<span class="meta">{parsed.sent_at:%Y-%m-%d %H:%M}</span></div>'
        f'<p class="subject">{_e(parsed.subject) or "(no subject)"}</p>'
        f'<div class="cols"><div><p class="label">raw body</p><pre>{_e(raw[:RAW_CAP])}</pre>{more}</div>'
        f'<div><p class="label">cleaned</p>{cleaned}</div></div></section>'
    )


def _e(text: str) -> str:
    return html.escape(text)


def _page(meta: list[tuple[str, str]], body: str) -> str:
    kv = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in meta)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mail cleaning, side by side</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
  --paper: #FBF8F3; --ink: #1C1A17; --ember: #C2410C; --muted: #97907F; --line: #E8E2D6;
  --font: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 13px/1.6 var(--font); }}
main {{ max-width: 1280px; margin: 0 auto; padding: 32px 16px 64px; }}
.tick {{ width: 32px; height: 4px; background: var(--ember); margin-bottom: 16px; }}
h1 {{ font-size: 20px; margin: 0; }}
h2 {{ font-size: 15px; margin: 40px 0 0; }}
h2 .n, .tag.hot {{ color: var(--ember); }}
hr {{ border: 0; border-top: 1px solid var(--line); margin: 8px 0 0; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 2px 24px; margin: 16px 0 0; }}
dt, .meta, .label, .tag, .head, .cut {{ color: var(--muted); }}
dd {{ margin: 0; }}
.card {{ border-bottom: 1px solid var(--line); padding: 20px 0; }}
.row {{ display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
.path {{ font-weight: 600; }}
.subject {{ margin: 2px 0 12px; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.cols > div {{ min-width: 0; }}
.label, .tag {{ margin: 0 0 4px; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
.tag {{ margin-top: 12px; }}
pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }}
.cut {{ text-decoration: line-through; }}
.head {{ font-size: 11px; }}
.quoted {{ border-left: 1px solid var(--line); padding-left: 12px; margin-top: 4px; }}
@media (max-width: 720px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><main>
<div class="tick"></div>
<h1>Mail cleaning, side by side</h1>
<div class="row meta"><span>rag-trend-radar · ingest/clean.py</span><span>{date.today():%Y-%m-%d}</span></div>
<dl>{kv}</dl>
{body}
</main></body></html>"""


if __name__ == "__main__":
    main(Path(sys.argv[1]))
