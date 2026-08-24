#!/bin/bash
# safe_run.sh — run a memory-hungry python script under a hard memory watchdog.
# Kills the child BEFORE macOS starts swap-thrashing (which freezes/crashes the machine).
#
# Usage: ./safe_run.sh script.py [args...]
# Tunables via env: RSS_LIMIT_MB, SWAP_GROWTH_MB, FREE_PCT_MIN, THREADS

set -uo pipefail

RSS_LIMIT_MB=${RSS_LIMIT_MB:-4200}     # kill child above this resident size
SWAP_GROWTH_MB=${SWAP_GROWTH_MB:-700}  # kill if swap grows this much over baseline
SWAP_MIN_RSS_MB=${SWAP_MIN_RSS_MB:-1500} # ...but only once the child is itself this big
FREE_PCT_MIN=${FREE_PCT_MIN:-8}        # kill if system free memory drops below this %
FREE_STRIKES=${FREE_STRIKES:-3}        # ...for this many consecutive samples
THREADS=${THREADS:-2}
POLL=${POLL:-1}

PY="$(dirname "$0")/.venv/bin/python"
LOG="${LOG:-/tmp/safe_run_$$.log}"

swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
free_pct()     { memory_pressure 2>/dev/null | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p' | tail -1; }

BASE_SWAP=$(swap_used_mb)
echo "[watchdog] baseline swap=${BASE_SWAP}MB  limits: rss<${RSS_LIMIT_MB}MB swap<+${SWAP_GROWTH_MB}MB (once rss>${SWAP_MIN_RSS_MB}MB) free>${FREE_PCT_MIN}%"

# Constrain thread pools so the CPU isn't saturated and memory arenas stay small.
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
       VECLIB_MAXIMUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
       CPU_NUM=$THREADS FLAGS_use_mkldnn=0 TOKENIZERS_PARALLELISM=false \
       PYTORCH_ENABLE_MPS_FALLBACK=1

nice -n 10 "$PY" "$@" > "$LOG" 2>&1 &
CHILD=$!
echo "[watchdog] pid=$CHILD log=$LOG"

REASON=""
PEAK=0
FREE_LOW=0
while kill -0 "$CHILD" 2>/dev/null; do
  # Sum RSS across the child and every descendant process.
  PIDS=$(pgrep -P "$CHILD" 2>/dev/null | tr '\n' ' ')
  RSS_KB=$(ps -o rss= -p "$CHILD" $PIDS 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ -z "$RSS_KB" ] || [ "$RSS_KB" = "0" ] && ! kill -0 "$CHILD" 2>/dev/null && break
  RSS_MB=$((RSS_KB / 1024))
  [ "$RSS_MB" -gt "$PEAK" ] && PEAK=$RSS_MB

  SW=$(swap_used_mb); SW_DELTA=$((SW - BASE_SWAP))
  FP=$(free_pct); FP=${FP:-100}

  if [ "$RSS_MB" -gt "$RSS_LIMIT_MB" ]; then REASON="RSS ${RSS_MB}MB > ${RSS_LIMIT_MB}MB"; fi
  # Swap is a system-wide number. On a machine already under pressure from other
  # apps it climbs on its own, so only blame the child once it is big enough to
  # plausibly be the cause -- otherwise a 200MB script gets killed for Chrome.
  if [ "$SW_DELTA" -gt "$SWAP_GROWTH_MB" ] && [ "$RSS_MB" -gt "$SWAP_MIN_RSS_MB" ]; then
    REASON="swap grew ${SW_DELTA}MB > ${SWAP_GROWTH_MB}MB while child held ${RSS_MB}MB"
  fi
  # Free memory is system-wide and dips transiently. Require several consecutive
  # violations and a child big enough to be worth blaming, so a brief dip caused
  # by another application does not kill a well-behaved job.
  if [ "$FP" -lt "$FREE_PCT_MIN" ] && [ "$RSS_MB" -gt "$SWAP_MIN_RSS_MB" ]; then
    FREE_LOW=$((FREE_LOW + 1))
    if [ "$FREE_LOW" -ge "$FREE_STRIKES" ]; then
      REASON="free memory ${FP}% < ${FREE_PCT_MIN}% for ${FREE_LOW} samples while child held ${RSS_MB}MB"
    fi
  else
    FREE_LOW=0
  fi

  if [ -n "$REASON" ]; then
    echo "[watchdog] TRIPPED: $REASON — killing $CHILD"
    pkill -9 -P "$CHILD" 2>/dev/null; kill -9 "$CHILD" 2>/dev/null
    wait "$CHILD" 2>/dev/null
    echo "[watchdog] peak RSS ${PEAK}MB. See $LOG"
    exit 99
  fi
  printf "[watchdog] rss=%sMB peak=%sMB swap+%sMB free=%s%%\n" "$RSS_MB" "$PEAK" "$SW_DELTA" "$FP"
  sleep "$POLL"
done

wait "$CHILD"; RC=$?
echo "[watchdog] child exited rc=$RC, peak RSS ${PEAK}MB. Log: $LOG"
exit $RC
