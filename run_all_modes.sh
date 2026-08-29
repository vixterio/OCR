#!/bin/bash
# Run one PDF through each of the six methods SEPARATELY, then compare.
#
#   ./run_all_modes.sh record.pdf [lang] [ram_gb]
#
# Each method is an independent run with its own output directory and its own VL
# model. No run uses more than one VL model: the +ocr methods add PP-OCRv6 and
# Tesseract as verifiers of that single model's numbers.
#
# The server holds one model at a time, so this restarts it between families.
# That is why the modes are ordered by family -- it avoids reloading a model.
set -uo pipefail
cd "$(dirname "$0")"

PDF="${1:?usage: ./run_all_modes.sh file.pdf [lang] [ram_gb]}"
LANG_ARG="${2:-eng}"
RAM_ARG="${3:-}"
COOLDOWN=${COOLDOWN:-45}   # seconds between modes

# Six VL model loads back-to-back is the heaviest sustained load this machine
# sees, and it is fanless. Check power once, up front, rather than discovering it
# at mode five.
if ! pmset -g batt 2>/dev/null | grep -q "AC Power"; then
  BATT=$(pmset -g batt 2>/dev/null | grep -oE "[0-9]+%" | head -1 | tr -d '%')
  echo "WARNING: on battery at ${BATT:-?}%. Six sequential VL model loads will drain it,"
  echo "         and this machine has already shut down mid-run once. Plug in first."
  echo "         Continuing in 10s; Ctrl-C to stop."
  sleep 10
fi

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT="output/compare-$STAMP"
mkdir -p "$ROOT"
echo "comparing six methods on $PDF -> $ROOT"

RAMOPT=()
[ -n "$RAM_ARG" ] && RAMOPT=(--ram-gb "$RAM_ARG")

for pair in "paddle:paddle" "paddle:paddle+ocr" \
            "deepseek:deepseek" "deepseek:deepseek+ocr" \
            "qwen:qwen" "qwen:qwen+ocr"; do
  FAMILY="${pair%%:*}"; MODE="${pair##*:}"
  SAFE="${MODE//+/_}"
  echo
  echo "================ $MODE ================"
  ./start_server.sh "$FAMILY" ${RAM_ARG:+$RAM_ARG} || { echo "server failed for $FAMILY"; continue; }
  RSS_LIMIT_MB=${RSS_LIMIT_MB:-3800} SWAP_GROWTH_MB=${SWAP_GROWTH_MB:-1500} \
  SWAP_MIN_RSS_MB=${SWAP_MIN_RSS_MB:-2200} FREE_PCT_MIN=${FREE_PCT_MIN:-4} \
  FREE_STRIKES=${FREE_STRIKES:-4} POLL=5 \
  LOG="$ROOT/$SAFE.log" \
  ./safe_run.sh run_ocr.py "$PDF" --mode "$MODE" --lang "$LANG_ARG" \
      --outdir "$ROOT/$SAFE" --allow-gaps "${RAMOPT[@]}" > "$ROOT/$SAFE.watchdog" 2>&1
  echo "  exit=$? peak=$(grep -o 'peak RSS [0-9]*MB' "$ROOT/$SAFE.watchdog" | tail -1)"
  # Let the SoC shed heat and let the previous model's memory actually be
  # released before loading the next multi-gigabyte one.
  pkill -f "vl_worker|mlx_vlm.server|mlx_server_patched" 2>/dev/null
  echo "  cooling down ${COOLDOWN}s before the next mode"
  sleep "$COOLDOWN"
  grep -E "^pages |^VL: |^wall " "$ROOT/$SAFE.log" 2>/dev/null | sed 's/^/  /'
done

echo
echo "================ comparison ================"
.venv/bin/python compare_modes.py "$ROOT"
