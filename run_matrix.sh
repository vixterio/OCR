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
  if [ -f "$ROOT/$label/audit.json" ]; then
    echo "---- $label (already done, skipping) ----"
    return 0
  fi
  echo "---- $label ----"
  ./start_server.sh "$fam" >/dev/null 2>&1
  local flag=""
  [ "$reads" = "on" ] && flag="--vl-line-reads"
  [ "$reads" = "off" ] && flag="--no-vl-line-reads"
  # FREE_PCT_MIN=3 with POLL=15 is what let a machine shutdown happen with the
  # watchdog running: free fell 14% -> 5% -> 4% between samples and every guard
  # was gated behind a child RSS that MLX never shows. Hard floor, fast polling.
  NICE=0 MIN_BATTERY_PCT=20 RSS_LIMIT_MB=3800 SWAP_GROWTH_MB=2500 \
  SWAP_MIN_RSS_MB=2400 FREE_PCT_MIN=8 FREE_STRIKES=3 \
  FREE_PCT_HARD=12 FREE_HARD_STRIKES=2 POLL=2 \
  LOG="$ROOT/$label.log" \
  ./safe_run.sh run_ocr.py "$PDF" --mode "$mode" --lang "$LANG_ARG" \
      --outdir "$ROOT/$label" --allow-gaps $flag > "$ROOT/$label.wd" 2>&1
  grep -E "^pages |^VL: |^wall " "$ROOT/$label.log" 2>/dev/null | sed 's/^/    /'
  grep -oE "TRIPPED: .*" "$ROOT/$label.wd" 2>/dev/null | tail -1 | sed 's/^/    watchdog /'
  # Release the model and let memory actually come back before the next run.
  pkill -f "vl_worker" 2>/dev/null
  pkill -f "mlx_vlm.server|mlx_server_patched" 2>/dev/null
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
