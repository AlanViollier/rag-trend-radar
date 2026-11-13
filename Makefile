.PHONY: help hooks format lint lint-slop typecheck test check ci audit scan-history backup-cache restore-cache check-investigator clean

help:
	@echo "Targets:"
	@echo "  hooks         - Install the pre-commit + pre-push hooks"
	@echo "  format        - ruff format, then ruff's safe fixes"
	@echo "  lint          - ruff check"
	@echo "  lint-slop     - Separator comments + the .keywords blocklist, over every tracked file"
	@echo "  typecheck     - mypy (strict)"
	@echo "  test          - pytest"
	@echo "  check         - Everything above but hooks and format; what pre-commit and CI run"
	@echo "  ci            - What GitHub CI runs: every pre-commit hook + pip-audit"
	@echo "  audit         - CVE scan of the locked dependencies (network)"
	@echo "  scan-history  - Secrets, .keywords and once-committed ignored files across every commit"
	@echo "  backup-cache  - Every computed vector to data/backups/ (19 hours of embedding, ~0.9 GB)"
	@echo "  restore-cache - Load the newest backup into an empty cache (after losing the volume)"
	@echo "  check-investigator - Play every run of page 2 in a headless browser, screenshots to data/"
	@echo "  clean         - Remove caches and build output"

hooks:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

# Ruff has no rule for decorative separator comments, so grep catches them in every
# tracked text file: a comment marker followed by a run of box-drawing characters, or
# by four or more `=`/`-`. Two dashes pass, so `# --flag` comments are fine.
# Second pass: .keywords (gitignored, one regex per line, `#` lines skipped) is a
# project-local blocklist: names, paths, emails. LICENSE names the author on purpose.
lint-slop:
	@result=$$(git ls-files -z | xargs -0 grep -nIE '^[[:space:]]*(#|//)+[[:space:]]*([─═]{2,}|[=-]{4,})' 2>/dev/null || true); \
	if [ -n "$$result" ]; then \
	  echo "  ✗ separator comments (use '# Section name'):"; echo "$$result" | sed 's/^/    /'; exit 1; \
	fi
	@if [ -f .keywords ]; then \
	  patterns=$$(grep -vE '^[[:space:]]*(#|$$)' .keywords | paste -sd '|' -); \
	  if [ -n "$$patterns" ]; then \
	    result=$$(git ls-files -z -- . ':!LICENSE' | xargs -0 grep -inIE "($$patterns)" 2>/dev/null || true); \
	    if [ -n "$$result" ]; then \
	      echo "  ✗ .keywords match:"; echo "$$result" | sed 's/^/    /'; exit 1; \
	    fi; \
	  fi; \
	fi

typecheck:
	uv run mypy .

# Exit code 5 is "no tests collected": a fresh skeleton is still green.
test:
	uv run pytest -rs || test $$? -eq 5

check:
	uv run ruff format --check .
	uv run ruff check .
	@$(MAKE) --no-print-directory lint-slop
	uv run mypy .
	@$(MAKE) --no-print-directory test

ci:
	uv run pre-commit run --all-files --show-diff-on-failure
	uv run pip-audit --skip-editable

audit:
	uv run pip-audit --skip-editable

# Page 2 played run by run in headless Chromium. Playwright isn't a project dependency: uv
# fetches it for this run only, pointed at the Chromium Playwright already downloaded (set
# CHROMIUM to use another one).
CHROMIUM ?= $(lastword $(wildcard $(HOME)/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell))
check-investigator:
	uv run --no-project --with playwright python scripts/check_investigator.py docs/investigator/index.html data/reports/investigator_shots "$(CHROMIUM)"

# Run before a repo goes public. Needs the gitleaks binary (`brew install gitleaks`). History is
# checked against the .keywords lines above `# files only`; lint-slop checks today's files against all.
scan-history:
	gitleaks git --redact --no-banner .
	@if [ -f .keywords ]; then \
	  patterns=$$(sed '/^# files only/q' .keywords | grep -vE '^[[:space:]]*(#|$$)' | paste -sd '|' -); \
	  result=$$(git log -p --all --format= -- . ':!LICENSE' | grep '^+' | grep -inE "($$patterns)" || true); \
	  if [ -n "$$result" ]; then echo "  ✗ .keywords match in history:"; echo "$$result" | sed 's/^/    /'; exit 1; fi; \
	fi
	@result=$$(git log --all --diff-filter=A --name-only --format= | sort -u | git check-ignore --no-index --stdin || true); \
	if [ -n "$$result" ]; then echo "  ✗ once committed, now ignored:"; echo "$$result" | sed 's/^/    /'; exit 1; fi
	@echo "  ✓ history clean"

# The embedding cache survives db-reset but not a deleted Docker volume. Everything else
# rebuilds from the maildir in about an hour; the vectors took about 19 hours.
backup-cache:
	@mkdir -p data/backups
	docker compose exec -T db pg_dump -U radar -d radar -t cache.embeddings -Fc > data/backups/embeddings-$$(date +%Y-%m-%d).dump
	@ls -lh data/backups/embeddings-$$(date +%Y-%m-%d).dump

restore-cache:
	docker compose exec -T db pg_restore -U radar -d radar --data-only -t embeddings < $$(ls -t data/backups/embeddings-*.dump | head -1)

clean:
	@rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis dist build *.egg-info
	@find . -not -path './.venv/*' -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
