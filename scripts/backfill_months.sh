#!/usr/bin/env bash
set -euo pipefail

# Backfill monthly forecasts (h6 + h12) by re-running the generator as-of given months.
# - Uses ASOF=YYYY-MM python generate_forecasts.py
# - Runs scripts/prepare_historical_predictions.py to write Historical_Predictions/*_h6.csv and *_h12.csv
# - Commits with a hardened git flow (autostash rebase) and pushes to main.
#
# Usage examples:
#   bash scripts/backfill_months.sh --months "2024-01,2024-02,2024-03"
#   bash scripts/backfill_months.sh --from 2024-01 --to 2025-12 --skip-existing-h12
#   bash scripts/backfill_months.sh --auto-missing-h12
#
# Requirements:
#   - Python and dependencies (pip install -r requirements.txt)
#   - UCDP_API_TOKEN set in environment (or configured in code) if required for data fetch

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

MONTHS=()
FROM=""
TO=""
SKIP_EXISTING_H12=false
AUTO_MISSING=false

function usage() {
  cat <<EOF
Backfill monthly forecasts by running the generator with ASOF.

Options:
  --months LIST           Comma-separated months (e.g., 2024-01,2024-02)
  --from YYYY-MM          First month (inclusive)
  --to YYYY-MM            Last month (inclusive)
  --skip-existing-h12     Skip months that already have an h12 snapshot
  --auto-missing-h12      Auto-detect months present without h12 and backfill them
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --months)
      IFS=',' read -ra MONTHS <<< "${2:-}"
      shift 2 ;;
    --from)
      FROM="${2:-}"; shift 2 ;;
    --to)
      TO="${2:-}"; shift 2 ;;
    --skip-existing-h12)
      SKIP_EXISTING_H12=true; shift ;;
    --auto-missing-h12)
      AUTO_MISSING=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

function month_inc() {
  local ym="$1"; local y=${ym%-*}; local m=${ym#*-}
  local mm=$((10#${m})); local yy=$((10#${y}))
  mm=$((mm+1)); if (( mm>12 )); then mm=1; yy=$((yy+1)); fi
  printf "%04d-%02d" "$yy" "$mm"
}

function build_range() {
  local a="$1"; local b="$2"; local cur="$a"; local out=()
  while :; do
    out+=("$cur")
    if [[ "$cur" == "$b" ]]; then break; fi
    cur=$(month_inc "$cur")
  done
  printf "%s\n" "${out[@]}"
}

function has_h12() {
  local p="$1"
  ls -1 Historical_Predictions 2>/dev/null | grep -E "^${p}(_|\.).*h12\.csv$" >/dev/null || return 1
}

function has_any_for_period() {
  local p="$1"
  ls -1 Historical_Predictions 2>/dev/null | grep -E "^${p}(_|\.).*\.csv$" >/dev/null || return 1
}

if $AUTO_MISSING; then
  # Detect months that exist in Historical_Predictions but lack h12
  mapfile -t detected < <(ls -1 Historical_Predictions 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}_' | sed -E 's/^([0-9]{4}-[0-9]{2}).*/\1/' | sort -u)
  MONTHS=()
  for m in "${detected[@]}"; do
    if ! has_h12 "$m"; then MONTHS+=("$m"); fi
  done
fi

if [[ -n "$FROM" && -n "$TO" ]]; then
  mapfile -t range < <(build_range "$FROM" "$TO")
  MONTHS=("${range[@]}")
fi

if [[ ${#MONTHS[@]} -eq 0 ]]; then
  echo "No months provided. Use --months or --from/--to or --auto-missing-h12." >&2
  exit 2
fi

echo "Backfilling months: ${MONTHS[*]}"

# Ensure repo is current
git fetch origin main
git reset --hard origin/main

# Configure bot identity (overrideable by Git config)
git config user.name  "pace-bot"
git config user.email "pace-bot@users.noreply.github.com"

# Ensure merge drivers are configured so rebases auto-accept generated artifacts
"$(dirname "$0")/ci-merge-driver-setup.sh" || true

for M in "${MONTHS[@]}"; do
  if $SKIP_EXISTING_H12 && has_h12 "$M"; then
    echo "[${M}] h12 already present, skipping (--skip-existing-h12)"
    continue
  fi

  echo "[${M}] Generating forecasts (ASOF=${M})"
  ASOF="$M" python3 generate_forecasts.py

  echo "[${M}] Preparing Historical_Predictions outputs"
  python3 scripts/prepare_historical_predictions.py

  # Stage only files for this period if present; otherwise stage all changes
  if has_any_for_period "$M"; then
    git add Historical_Predictions/${M}*.csv || true
  else
    git add -A
  fi

  if git diff --cached --quiet; then
    echo "[${M}] No changes to commit"
  else
    git commit -m "chore(archive): add ${M} h6/h12 snapshots to Historical_Predictions (backfill)"
  fi

  # Rebase with autostash to avoid failures due to local changes
  git -c rebase.autoStash=true pull --rebase origin main || {
    echo "[${M}] Warning: pull --rebase failed, retrying once" >&2
    git -c rebase.autoStash=true pull --rebase origin main
  }

  # Final combine and push
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "chore(archive): finalize ${M} (post-rebase)"
  fi
  git push origin HEAD:main
  echo "[${M}] ✓ Pushed"
done

echo "All requested months processed."
