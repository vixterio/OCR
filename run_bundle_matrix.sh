#!/bin/bash
# Run every mode over the same pages of one bundle, one at a time.
#
# Sequential by design. These are 8GB-class machines and the failure mode is not
# a slow run, it is the laptop shutting down -- so modes never overlap, the
# watchdog supervises each one, and families are grouped so the server reloads a
# model three times rather than eight.
set -u
PDF=${PDF:-merged_bundles/bundle_001.pdf}
N=${N:-20}
LANG_T=${LANG_T:-deu}          # these bundles are German
TAG=${TAG:-b1}
OUT=output/$TAG
mkdir -p "$OUT"

run () {                        # run <mode> <family>
  # Declared on separate lines deliberately. `local a=$1 b="${a/x/y}"` declares
  # every name first and only then assigns, so the expansion of $mode saw an
  # empty variable and every mode wrote to the same directory, silently
  # overwriting the previous one's audit.
  local mode=$1
  local fam=$2
  local dir="$OUT/${mode/+/_}"
  if [ -f "$dir/audit.json" ] && [ -z "${FORCE:-}" ]; then
    echo "=== $mode: already done, skipping (FORCE=1 to redo) ==="
    return
  fi
  echo "=== $mode ($(date +%H:%M:%S)) ==="
  if [ "$fam" != "deepseek" ] && [ "$fam" != "$LOADED" ]; then
    ./start_server.sh "$fam" >/dev/null 2>&1 && LOADED=$fam
  fi
  NICE=0 SWAP_GROWTH_MB=3000 ./safe_run.sh run_ocr.py "$PDF" \
      --mode "$mode" --max-pages "$N" --lang "$LANG_T" --allow-gaps \
      --outdir "$dir" > "$OUT/$mode.out" 2>&1
  echo "  rc=$? $(grep -oE 'peak RSS.*' "$OUT/$mode.out" | tail -1)"
}

LOADED=""
for pair in "granite granite" "granite+ocr granite" \
            "paddle paddle" "paddle+ocr paddle" \
            "qwen qwen" "qwen+ocr qwen" \
            "deepseek deepseek" "deepseek+ocr deepseek"; do
  run $pair
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
