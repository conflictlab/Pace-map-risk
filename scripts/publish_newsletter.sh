#!/usr/bin/env bash
set -euo pipefail

# Robust publisher for newsletter artifacts.
# - Generates images and PDF (optional if pre-generated)
# - Commits changes
# - Rebases onto origin/main with autostash
# - Auto-resolves binary conflicts for PDFs by keeping the rebased commit (theirs)

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

git fetch origin main

# Optional generation steps (uncomment if you want to regenerate)
# python3 scripts/generate_best_from_forecasts.py || true
# python3 scripts/newsletter_images_from_forecasts.py || true
# python3 pdf.py || true

# Stage newsletter outputs
git add assets/Report.pdf Newsletters/*.pdf assets/html_output/_pdf_images || true

if git diff --cached --quiet; then
  echo "No newsletter changes to commit"
else
  git commit -m "chore(newsletter): publish $(date +"%B %Y") newsletter (PDF/HTML)"
fi

# Configure merge drivers for auto-resolution
"$ROOT_DIR/scripts/ci-merge-driver-setup.sh" || true

# Rebase with autostash; if conflicts remain, prefer rebased commit for PDFs
if ! git -c rebase.autoStash=true pull --rebase origin main; then
  echo "Rebase failed; resolving newsletter binary conflicts by preferring rebased commit (theirs)" >&2
  if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
    # Resolve known generated artifacts
    for f in $(git diff --name-only --diff-filter=U); do
      case "$f" in
        assets/Report.pdf|Newsletters/*.pdf|assets/html_output/_pdf_images/*)
          git checkout --theirs -- "$f" || true ;;
        *) ;;
      esac
    done
    git add -A
    git rebase --continue || {
      echo "Rebase continue failed; aborting" >&2
      git rebase --abort || true
      exit 1
    }
  else
    echo "Not in rebase state; aborting" >&2
    exit 1
  fi
fi

git push origin HEAD:main
echo "✓ Newsletter publish completed"

