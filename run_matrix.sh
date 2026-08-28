#!/bin/bash
# Run the full mode x line-reads matrix on one document and score every run
# against ground truth. Ordered by model family so each model loads once.
set -uo pipefail
cd "$(dirname "$0")"
PDF="${1:-test_record.pdf}"
LANG_ARG="${2:-script/Latin}"
ROOT="output/matrix"
COOLDOWN="${COOLDOWN:-20}"
mkdir -p "$ROOT"

run() {   # family mode reads label
  local fam="$1" mode="$2" reads="$3" label="$4"
  echo "---- $label ----"
  ./start_server.sh "$fam" >/dev/null 2>&1
  local flag=""
  [ "$reads" = "on" ] && flag="--vl-line-reads"
  [ "$reads" = "off" ] && flag="--no-vl-line-reads"
  NICE=0 MIN_BATTERY_PCT=20 RSS_LIMIT_MB=3800 SWAP_GROWTH_MB=2500 \
  SWAP_MIN_RSS_MB=2400 FREE_PCT_MIN=3 FREE_STRIKES=5 POLL=15 \
  LOG="$ROOT/$label.log" \
  ./safe_run.sh run_ocr.py "$PDF" --mode "$mode" --lang "$LANG_ARG" \
      --outdir "$ROOT/$label" --allow-gaps $flag > "$ROOT/$label.wd" 2>&1
  grep -E "^pages |^VL: |^wall " "$ROOT/$label.log" 2>/dev/null | sed 's/^/    /'
  pkill -f "vl_worker" 2>/dev/null
  sleep "$COOLDOWN"
}

for fam in paddle deepseek qwen granite; do
  run "$fam" "$fam"        na  "${fam}"
  run "$fam" "${fam}+ocr"  on  "${fam}_ocr_on"
  run "$fam" "${fam}+ocr"  off "${fam}_ocr_off"
done

echo
echo "================ scored against ground truth ================"
.venv/bin/python evaluate.py "$ROOT"/*/audit.json --table
