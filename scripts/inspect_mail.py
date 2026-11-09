"""Profile the Enron bodies before and after cleaning; prints a markdown report.

Usage: uv run python scripts/inspect_mail.py data/enron/maildir > data/reports/mail.md
"""

import hashlib
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_trend_radar.exceptions import ParseError
from rag_trend_radar.ingest.clean import normalize, split_quoted, strip_boilerplate
from rag_trend_radar.ingest.dedup import content_hash, resolve_sent_at
from rag_trend_radar.schema import ParsedEmail, RawFile, SplitBody
from rag_trend_radar.sources import enron

# Each detector is one kind of leftover. A hit means the text still carries it.
DETECTORS = {
    name: re.compile(pattern, re.MULTILINE)
    for name, pattern in {
        "original message": r"^\s*-{2,}\s*Original Message\s*-{2,}",
        "forwarded by": r"^\s*-{3,}\s*Forwarded by",
        "quoted header block": r"^\s*From:.*\n\s*(?:Sent|Date):.*\n\s*To:",
        "lotus header": r"^.+@(?:ECT|ENRON|EES|ENRON_DEVELOPMENT)\s*\n\s*\d{2}/\d{2}/\d{2,4} \d{2}:\d{2}",
        "on … wrote": r"^On .{5,100}wrote:\s*$",
        "> quoted line": r"^>",
        "header field line": r"^\s*(?:From|To|Sent|cc|Subject)\s*:",
        "html tag": r"(?i)</?(?:html|body|div|p|br|font|table|tr|td|span)\b[^>\n]*>",
        "html entity": r"&(?:nbsp|amp|lt|gt|quot|#\d{2,4});",
        "quoted-printable residue": r"=(?:20|09|3D)$",
        "attachment stub": r"<<\s*(?:File:\s*)?[^<>\n]+\.(?:doc|xls|pdf|ppt|txt|zip)\s*>>",
        "mime part": r"^(?:------_=_NextPart|Content-Type:|Content-Transfer-Encoding:)",
        "base64 run": r"^(?:[A-Za-z0-9+/]{60,}\n){3,}",
        "disclaimer": r"(?is)(?:confidential|privileged).{0,300}(?:intended recipient|received this (?:e-?mail|message) in error)",
        "separator line": r"^[*~=_\-]{10,}\s*$",
    }.items()
}
THRESHOLDS = (50, 100, 200, 500)  # boilerplate candidates: a line in at least this many emails
TOP_LINES, TOP_LONGEST, SAMPLE = 30, 20, 20


@dataclass(frozen=True)
class Profile:
    """What pass 1 keeps per file: enough to count, never the body itself."""

    path: str
    key: str
    strict_key: str
    sent_at: datetime
    chars: int
    own_chars: int
    raw_hits: frozenset[str]
    own_hits: frozenset[str]
    own_hash: str
    block_hashes: tuple[str, ...]
    first_block: str
    own_lines: bytes


def main(root: Path) -> None:
    """Profile every file under `root`, dedup, then print the report."""
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    quarantine: Counter[str] = Counter()
    by_key: dict[str, Profile] = {}
    dates: dict[str, list[datetime]] = {}
    strict_keys: set[str] = set()
    with ProcessPoolExecutor() as pool:
        for result in pool.map(_profile, [(root, p) for p in paths], chunksize=2000):
            if isinstance(result, str):
                quarantine[result] += 1
                continue
            strict_keys.add(result.strict_key)
            dates.setdefault(result.key, []).append(result.sent_at)
            by_key.setdefault(result.key, result)
    unique = list(by_key.values())

    line_counts: Counter[int] = Counter()
    for p in unique:
        line_counts.update(_unpack(p.own_lines))
    wanted = {h for h, n in line_counts.items() if n >= min(THRESHOLDS)}
    texts = _line_texts(root, unique, wanted)
    counts = {texts[h]: line_counts[h] for h in wanted if h in texts}
    with ProcessPoolExecutor(initializer=_init_trim, initargs=(counts,)) as pool:
        trims = list(pool.map(_trim, [(root, p.path) for p in unique], chunksize=2000))

    out = _Report()
    out.section("01 · Corpus")
    out.kv("files", len(paths))
    out.kv("quarantined", ", ".join(f"{r} {n}" for r, n in quarantine.most_common()))
    out.kv("unique, exact (sender, date, subject, body)", len(strict_keys))
    out.kv("unique, content_hash (subject, body)", len(unique))
    copies = Counter(min(len(d), 5) for d in dates.values())
    out.kv("copies per email (1/2/3/4/5+)", " / ".join(str(copies[n]) for n in range(1, 6)))

    out.section("02 · Dates (resolve_sent_at)")
    outcomes = Counter(_date_outcome(d) for d in dates.values())
    for outcome, n in outcomes.most_common():
        out.kv(outcome, f"{n} ({_pct(n, len(unique))})")

    out.section("03 · Length (chars, unique emails)")
    for label, values in (
        ("body", [p.chars for p in unique]),
        ("own", [p.own_chars for p in unique]),
    ):
        q = statistics.quantiles(values, n=100)
        out.kv(
            f"{label} p50 / p90 / p99 / max",
            f"{q[49]:.0f} / {q[89]:.0f} / {q[98]:.0f} / {max(values)}",
        )

    out.section("04 · Detectors: raw body vs own text")
    out.table(["detector", "raw", "own", "own examples"])
    for name in DETECTORS:
        raw = sum(name in p.raw_hits for p in unique)
        own = [p for p in unique if name in p.own_hits]
        examples = " ".join(p.path for p in own[:3])
        out.row(
            name, _pct(raw, len(unique)), f"{_pct(len(own), len(unique))} ({len(own)})", examples
        )

    out.section("05 · Split")
    own_hashes = {p.own_hash for p in unique if p.own_chars}
    blocks = [h for p in unique for h in p.block_hashes]
    with_blocks = sum(bool(p.block_hashes) for p in unique)
    matched = sum(h in own_hashes for h in blocks)
    out.kv("emails with quoted messages", f"{with_blocks} ({_pct(with_blocks, len(unique))})")
    out.kv("quoted blocks", len(blocks))
    out.kv(
        "blocks whose text is another email's own text", f"{matched} ({_pct(matched, len(blocks))})"
    )
    orphans = {h for h in blocks if h not in own_hashes}
    out.kv("orphan blocks", len(blocks) - matched)
    out.kv("distinct orphan texts (what rule A would chunk)", len(orphans))
    empty = Counter(p.first_block for p in unique if not p.own_chars)
    out.kv(
        "own text empty, by what opens the first block",
        ", ".join(f"{k} {n}" for k, n in empty.most_common()),
    )
    kept = statistics.quantiles([p.own_chars / p.chars for p in unique if p.chars], n=10)
    out.kv("own / body chars, deciles", " ".join(f"{k:.2f}" for k in kept))
    suspicious = [p for p in unique if not p.own_chars and p.first_block not in ("forward", "none")]
    out.kv("empty own, not a forward (read these)", " ".join(p.path for p in suspicious[:10]))

    out.section("06 · Boilerplate trim by threshold")
    out.table(
        ["min emails", "lines", "emails trimmed", "own chars trimmed", "kept whole by the guard"]
    )
    own_total = sum(p.own_chars for p in unique)
    for i, threshold in enumerate(THRESHOLDS):
        trimmed = [t[i] for t in trims]
        out.row(
            threshold,
            sum(n >= threshold for n in counts.values()),
            _pct(sum(c > 0 for c, _ in trimmed), len(unique)),
            _pct(sum(c for c, _ in trimmed), own_total),
            sum(g for _, g in trimmed),
        )
    rng = random.Random(0)  # noqa: S311 (a reproducible sample, not a secret)
    for low, high in ((90, 110), (180, 220)):
        near = [(n, line) for line, n in counts.items() if low <= n <= high]
        out.lines.append(f"\nLines in {low}–{high} emails, just around a threshold:\n")
        out.lines += [
            f"- {n} · `{_cell(line, 90)}`"
            for n, line in sorted(rng.sample(near, min(15, len(near))), reverse=True)
        ]

    out.section(f"07 · Top {TOP_LINES} repeated lines in own text")
    out.table(["emails", "line"])
    for line, n in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_LINES]:
        out.row(n, _cell(line))

    longest = sorted(unique, key=lambda p: p.own_chars, reverse=True)[:TOP_LONGEST]
    sample = rng.sample(unique, SAMPLE)
    for title, picked in (
        (f"08 · Longest {TOP_LONGEST} by own text", longest),
        (f"09 · Random {SAMPLE}", sample),
    ):
        out.section(title)
        out.table(["path", "body", "own", "blocks", "own detectors", "first own line"])
        for p in picked:
            first = _cell(_first_own_line(root, p.path))
            out.row(
                p.path,
                p.chars,
                p.own_chars,
                len(p.block_hashes),
                ", ".join(sorted(p.own_hits)),
                first,
            )
    print(out.text())


# Pass 1 runs in worker processes: parse, clean, detect, hash.


def _profile(job: tuple[Path, Path]) -> Profile | str:
    root, path = job
    rel = path.relative_to(root).as_posix()
    try:
        parsed = _parse(root, rel)
    except ParseError as error:
        return str(error).split(":")[0]
    split = _split(parsed)
    return Profile(
        path=rel,
        key=content_hash(parsed.subject, parsed.body),
        strict_key=_digest(parsed.sender, parsed.sent_at.isoformat(), parsed.subject, parsed.body),
        sent_at=parsed.sent_at,
        chars=len(parsed.body),
        own_chars=len(split.own),
        raw_hits=_hits(parsed.body),
        own_hits=_hits(split.own),
        own_hash=_collapsed_hash(split.own),
        block_hashes=tuple(_collapsed_hash(b.text) for b in split.quoted if b.text),
        first_block=_first_block(split),
        own_lines=b"".join(_line_hash(line).to_bytes(8) for line in _lines(split.own)),
    )


def _parse(root: Path, rel: str) -> ParsedEmail:
    mailbox, *folders, _name = rel.split("/")
    raw = RawFile(
        path=rel, mailbox=mailbox, folder="/".join(folders), content=(root / rel).read_bytes()
    )
    return enron.parse(raw)


def _split(parsed: ParsedEmail) -> SplitBody:
    return split_quoted(normalize(parsed.body))


def _hits(text: str) -> frozenset[str]:
    return frozenset(name for name, rx in DETECTORS.items() if rx.search(text))


def _first_block(split: SplitBody) -> str:
    if not split.quoted:
        return "none"
    header = split.quoted[0].header
    for kind, marker in (
        ("forward", "Forwarded by"),
        ("outlook", "Original Message"),
        ("on-wrote", "wrote:"),
    ):
        if marker in header:
            return kind
    return "header" if header else "inline >"


def _lines(text: str) -> set[str]:
    return {key for line in text.split("\n") if (key := " ".join(line.split()))}


def _line_hash(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode(), digest_size=8).digest())


def _unpack(blob: bytes) -> Iterator[int]:
    return (int.from_bytes(blob[i : i + 8]) for i in range(0, len(blob), 8))


def _collapsed_hash(text: str) -> str:
    return _digest(" ".join(text.split()))


def _digest(*parts: str) -> str:
    return hashlib.blake2b("\x00".join(parts).encode(), digest_size=16).hexdigest()


def _date_outcome(copies: list[datetime]) -> str:
    _sent_at, uncertain = resolve_sent_at(copies)
    if uncertain:
        return "copies disagree, earliest kept, flagged uncertain"
    if len(set(copies)) > 1:
        return "copies disagree, the server-stamped one wins"
    return "one date"


# Pass 2: recover line texts from hashes, then trim boilerplate at each threshold.


def _line_texts(root: Path, unique: list[Profile], wanted: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for profile in unique:
        hits = wanted.intersection(_unpack(profile.own_lines)) - found.keys()
        if not hits:
            continue
        for line in _lines(_split(_parse(root, profile.path)).own):
            if (h := _line_hash(line)) in hits:
                found[h] = line
        if len(found) == len(wanted):
            break
    return found


_FREQUENT: list[frozenset[str]] = []


def _init_trim(counts: dict[str, int]) -> None:
    # frequent_lines() would recount the corpus; the counts are already here, so build the sets.
    _FREQUENT.extend(frozenset(line for line, n in counts.items() if n >= t) for t in THRESHOLDS)


def _trim(job: tuple[Path, str]) -> list[tuple[int, int]]:
    # Per threshold: chars trimmed, and 1 when every line was frequent and the guard kept it whole.
    root, rel = job
    own = _split(_parse(root, rel)).own
    keys = _lines(own)
    return [
        (len(own) - len(strip_boilerplate(own, f)), int(bool(keys) and keys <= f))
        for f in _FREQUENT
    ]


def _first_own_line(root: Path, rel: str) -> str:
    return next((line for line in _split(_parse(root, rel)).own.split("\n") if line.strip()), "")


# Markdown output.


class _Report:
    def __init__(self) -> None:
        self.lines: list[str] = ["# Enron mail inspection", ""]

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def kv(self, key: str, value: object) -> None:
        self.lines.append(f"- {key}: **{value}**")

    def table(self, header: list[str]) -> None:
        self.lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def row(self, *cells: object) -> None:
        self.lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    def text(self) -> str:
        return "\n".join(self.lines)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "–"


def _cell(text: str, width: int = 100) -> str:
    return text[:width].replace("|", "\\|").replace("`", "'")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
