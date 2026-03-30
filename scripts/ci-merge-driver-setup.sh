#!/usr/bin/env bash
set -euo pipefail

# Configure merge drivers used by CI to auto-resolve generated binary conflicts.
# We define an explicit driver that overwrites the working copy with the
# rebased commit's file ("theirs" during rebase), which is what we want for
# generated newsletter assets.

# keepTheirs: use the rebased commit's content (third arg: %B)
git config merge.keepTheirs.name "Keep theirs (rebased commit)"
git config merge.keepTheirs.driver 'sh -c '\''cat "$3" > "$2"'\'' _

# keepOurs: noop driver that keeps current version if ever needed elsewhere
git config merge.keepOurs.name "Keep ours"
git config merge.keepOurs.driver true

echo "Configured merge drivers:"
git config --get-regexp '^merge\..*\.driver' || true
