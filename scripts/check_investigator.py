"""Plays every run of page 2 in a headless browser and fails on anything missing or broken.

Each run is skipped to its end: every tool call, note and thought must be on the page, the answer
whole, every cited email named, no step left running, no "undefined". Screenshots land in `shots`
for a look by eye. Needs Playwright's Chromium, not a project dependency:

    make check-investigator
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHECK = """() => {
  const r = current, ans = document.querySelector('.thread .answer');
  const want = r.answer.replace(/^\\s*[-*] /gm, '').replace(/[*`#]/g, '').replace(/\\s+/g, '');
  const got = (ans ? ans.innerText : '').replace(/→/g, '').replace(/\\s+/g, '');
  const thoughts = [], pending = [];
  for (const s of r.steps) {
    if (s.kind === 'think') pending.push(s.text);
    if (s.kind === 'turn') { if (pending.length) thoughts.push(pending.join('\\n\\n')); pending.length = 0; }
  }
  const docs = [...new Set(r.citations.map(c => c.document_id))];
  return {
    tools: [document.querySelectorAll('.work .tool').length, r.steps.filter(s => s.kind === 'tool').length],
    notes: [document.querySelectorAll('.work .note').length, r.steps.filter(s => s.kind === 'note').length],
    thoughts: [[...document.querySelectorAll('.work .think .text')].map(e => e.textContent).join(''), thoughts.join('')],
    answer: [got, want],
    sources: [document.querySelectorAll('.src').length, docs.length],
    unnamed: docs.filter(d => !DOCS[d]).length,
    targets: r.citations.filter((c, j) => !document.getElementById('p-' + j)).length,
    running: document.querySelectorAll('.stage.live, .shimmer').length,
    junk: (document.body.innerText.match(/undefined|NaN|\\[object/g) || []).length,
    baseline: !!r.baseline.found,
  };
}"""


def main(page: Path, shots: Path, chromium: Path) -> int:
    """Check every run of `page` with the browser at `chromium`; 1 when any run failed."""
    shots.mkdir(parents=True, exist_ok=True)
    failed = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(chromium))
        tab = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        tab.on("pageerror", lambda e: errors.append(str(e)))
        tab.goto(page.resolve().as_uri())
        tab.wait_for_timeout(2500)
        tab.screenshot(path=str(shots / "00-playing.png"))
        ids = tab.eval_on_selector_all(".card", "cards => cards.map(c => c.dataset.id)")
        for n, run in enumerate(ids, 1):
            errors.clear()
            tab.click(f'.card[data-id="{run}"]')
            tab.wait_for_timeout(200)
            tab.click("#skip")
            tab.wait_for_selector(".meta", timeout=8000)
            tab.wait_for_timeout(300)
            seen = tab.evaluate(CHECK)
            wrong = [
                k
                for k in ("tools", "notes", "thoughts", "answer", "sources")
                if seen[k][0] != seen[k][1]
            ]
            wrong += [k for k in ("unnamed", "targets", "running", "junk") if seen[k]]
            wrong += [] if seen["baseline"] else ["baseline"]
            wrong += [f"error: {e}" for e in errors]
            failed += bool(wrong)
            sys.stdout.write(f"{run:26} {'ok' if not wrong else 'FAILED ' + ', '.join(wrong)}\n")
            tab.screenshot(path=str(shots / f"{n:02d}-{run}.png"))
        browser.close()
    sys.stdout.write(f"{len(ids) - failed} of {len(ids)} runs ok; screenshots in {shots}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
