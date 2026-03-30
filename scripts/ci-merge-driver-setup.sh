#!/usr/bin/env bash
set -euo pipefail

# Configure merge drivers used by CI to auto-resolve generated binary conflicts.
# - 'theirs' picks the rebased commit's content during rebase/cherry-pick
# - 'ours' is kept for completeness if some jobs rely on it

git config merge.theirs.driver true
git config merge.ours.driver true

echo "Configured merge drivers:"
git config --get-regexp '^merge\..*\.driver' || true

