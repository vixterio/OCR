#!/bin/bash
# Run one round's documents through each mode, one document at a time.
#
# Per-document rather than per-bundle: the generator produced one HTML file per
# document, so a bundle-wide transcript has no boundary to compare against. One
# PDF in, one transcript out, one source HTML to diff it against.
#
# Grouped by mode so the server loads each model once instead of once per
# document, and strictly sequential because the failure mode on this hardware is
# not a slow run, it is the machine becoming unresponsive.
set -u
ROUND=${ROUND:-work/round1}
MODES=${MODES:-"granite paddle qwen"}
LANG_T=${LANG_T:-deu}
OUT=${OUT:-work/out/$(basename "$ROUND")}
mkdir -p "$OUT"

# The absolute swap ceiling is raised above this machine's stale baseline: macOS
# does not shrink swap files without a reboot, so 5.4GB of accounting from
# earlier runs would refuse every start. The guard that actually prevents the
# WindowServer starvation is the free-memory hard floor, which is left on and
# has been verified to fire.
export SWAP_ABS_MB=${SWAP_ABS_MB:-7500}
export FREE_PCT_HARD=${FREE_PCT_HARD:-15}
export FREE_HARD_STRIKES=${FREE_HARD_STRIKES:-3}
export NICE=0

# name<TAB>path, because the documents now live in the generator's own output
# tree rather than beside the index, and several cases share a document_id.
DOCS=$(.venv/bin/python -c "
import json
for r in json.load(open('$ROUND/index.json')):
    print(f\"case{r['case_id']}_{r['document_id']}\t{r['pdf']}\")
")

for mode in $MODES; do
  fam=${mode%%+*}
  echo "=== loading $fam ($(date +%H:%M:%S)) ==="
  ./start_server.sh "$fam" >/dev/null 2>&1 || echo "  (server start reported a problem)"
  while IFS=$'\t' read -r doc pdf; do
    [ -z "$doc" ] && continue
    dir="$OUT/${mode/+/_}/$doc"
    if [ -f "$dir/audit.json" ] && [ -z "${FORCE:-}" ]; then
      echo "  $mode $doc: done, skipping"; continue
    fi
    mkdir -p "$dir"
    echo "  $mode $doc ($(date +%H:%M:%S))"
    ./safe_run.sh run_ocr.py "$pdf" --mode "$mode" --lang "$LANG_T" \
        --allow-gaps --outdir "$dir" > "$dir/run.out" 2>&1
    rc=$?
    echo "     rc=$rc $(grep -oE 'peak RSS.*' "$dir/run.out" | tail -1)"
    [ "$rc" != 0 ] && grep -E "TRIPPED|REFUSING" "$dir/run.out" | tail -1
  done <<< "$DOCS"
done
echo "=== ROUND DONE $(date +%H:%M:%S) ==="
