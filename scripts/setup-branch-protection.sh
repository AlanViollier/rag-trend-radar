#!/usr/bin/env bash
# Setup GitHub branch protection on `main` via Repository Rulesets.
# Requires the CI `check` job to pass before any merge into main.
#
# IMPORTANT: GitHub branch protection (both legacy API and Rulesets) is a
# PAID feature for private repos. This script fails soft with a clear message
# if your repo is private + free plan. Your options:
#   • Upgrade to GitHub Pro (~$4/mo): https://github.com/account/upgrade
#   • Make the repo public:           gh repo edit --visibility public
#   • Skip — local hooks (`make hooks`) + CI on push still enforce quality.
#
# Prerequisites:
#   - gh CLI installed and authenticated (`gh auth login`)
#   - You must be the repo owner or an admin
#   - `origin` remote must point at the GitHub repo
#
# Usage:
#   make protect
#
# What it enforces (when supported):
#   - CI 'check' job must pass before merge
#   - Block branch deletion
#   - Block force-pushes to main
#
# Status check context "check" matches the job ID in .github/workflows/ci.yml.
# Rename the workflow job → update the JSON body below too.

set -euo pipefail

if ! command -v gh > /dev/null 2>&1; then
  echo "  ✗ gh CLI not installed."
  echo "    Install: brew install gh   (or see https://cli.github.com)"
  exit 1
fi

if ! gh auth status > /dev/null 2>&1; then
  echo "  ✗ gh CLI not authenticated."
  echo "    Run: gh auth login"
  exit 1
fi

REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || true)
if [ -z "$REMOTE_URL" ]; then
  echo "  ✗ No 'origin' remote set."
  echo "    Add one: git remote add origin git@github.com:<owner>/<repo>.git"
  exit 1
fi

REPO_PATH=$(echo "$REMOTE_URL" | sed -E 's|.*[:/]([^/]+/[^/]+)\.git|\1|' | sed -E 's|.*[:/]([^/]+/[^/]+)$|\1|')
RULESET_NAME="Require CI on main"

# Pre-flight: can we even hit the rulesets API on this repo?
PREFLIGHT=$(gh api "repos/$REPO_PATH/rulesets" 2>&1) || PREFLIGHT_FAILED=true

if [ "${PREFLIGHT_FAILED:-false}" = "true" ]; then
  if echo "$PREFLIGHT" | grep -qi 'upgrade.*pro\|make this repository public'; then
    echo ""
    echo "  ⚠ Branch protection isn't available on this repo."
    echo "    GitHub requires Pro for private-repo branch protection (rulesets + legacy)."
    echo ""
    echo "  Options:"
    echo "    • Upgrade to GitHub Pro (~\$4/mo): https://github.com/account/upgrade"
    echo "    • Make the repo public:            gh repo edit --visibility public"
    echo "    • Skip — local hooks (\`make hooks\`) + CI on push still enforce quality."
    echo ""
    exit 0
  fi
  echo ""
  echo "  ✗ gh api failed:"
  echo "$PREFLIGHT" | sed 's/^/    /'
  echo ""
  exit 1
fi

echo ""
echo "  Setting ruleset on: $REPO_PATH"
echo "    - CI 'check' job must pass before merge"
echo "    - Block branch deletion + force-push on main"
echo ""

# Remove any existing ruleset with the same name (idempotent re-runs)
EXISTING_ID=$(gh api "repos/$REPO_PATH/rulesets" --jq ".[] | select(.name == \"$RULESET_NAME\") | .id" 2>/dev/null || true)
if [ -n "$EXISTING_ID" ]; then
  echo "  Removing existing ruleset (id: $EXISTING_ID)..."
  gh api -X DELETE "repos/$REPO_PATH/rulesets/$EXISTING_ID" > /dev/null
fi

# Create the new ruleset
gh api -X POST "repos/$REPO_PATH/rulesets" --input - > /dev/null <<EOF
{
  "name": "$RULESET_NAME",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {
      "type": "required_status_checks",
      "parameters": {
        "required_status_checks": [{"context": "check"}],
        "strict_required_status_checks_policy": false
      }
    }
  ]
}
EOF

echo "  ✓ Ruleset '$RULESET_NAME' active on $REPO_PATH"
echo ""
