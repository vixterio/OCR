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
# Priority of the child. Metal command buffers have an execution deadline, and a
# niced process cannot always feed the GPU fast enough to meet it: DeepSeek-OCR-2
# failed with "[METAL] Command buffer execution failed: GPU Timeout" under
# nice 10 and succeeded at normal priority on the same input. So GPU-bound work
# runs at NICE=0 by default; raise it for CPU-only jobs if you want the machine
# to stay responsive.
#
# Note the interaction with the power guards below: NICE=0 means the run takes
# the machine's full attention, which on a fanless Air on battery is how it ends
# up switching off. The two settings are a pair -- if you raise NICE to be kind
# to the machine, the Metal deadline may bite; if you keep NICE=0, plug in.
NICE=${NICE:-0}
POLL=${POLL:-1}
# Power guards. This is a fanless MacBook Air M2: sustained VL inference is the
# heaviest sustained load it ever sees, and on battery it will flatten the pack
# and the machine will simply switch off mid-run. Losing a run is cheap; losing
# the machine mid-write is not, and a half-written audit file is exactly what the
# atomic writes exist to prevent.
MIN_BATTERY_PCT=${MIN_BATTERY_PCT:-40}   # refuse to start on battery below this
ABORT_BATTERY_PCT=${ABORT_BATTERY_PCT:-15}  # stop the child if it drops this low
REQUIRE_AC=${REQUIRE_AC:-0}              # 1 = refuse to run unless plugged in
THERMAL_MIN=${THERMAL_MIN:-40}           # CPU_Speed_Limit % below which we stop

PY="$(dirname "$0")/.venv/bin/python"
LOG="${LOG:-/tmp/safe_run_$$.log}"

on_ac()      { pmset -g batt 2>/dev/null | grep -q "AC Power"; }
battery_pct() { pmset -g batt 2>/dev/null | grep -oE "[0-9]+%" | head -1 | tr -d '%'; }
speed_limit() { pmset -g therm 2>/dev/null | grep -oE "CPU_Speed_Limit[[:space:]]*=[[:space:]]*[0-9]+" | grep -oE "[0-9]+$"; }

swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
free_pct()     { memory_pressure 2>/dev/null | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p' | tail -1; }

# ---- pre-flight power check -------------------------------------------------
BATT=$(battery_pct); BATT=${BATT:-100}
if on_ac; then
  echo "[watchdog] power: AC connected (battery ${BATT}%)"
else
  if [ "$REQUIRE_AC" = "1" ]; then
    echo "[watchdog] REFUSING: REQUIRE_AC=1 and the machine is on battery power."
    echo "[watchdog] Plug in and retry. VL inference on a fanless Air will drain the pack."
    exit 3
  fi
  if [ "$BATT" -lt "$MIN_BATTERY_PCT" ]; then
    echo "[watchdog] REFUSING: on battery at ${BATT}%, below MIN_BATTERY_PCT=${MIN_BATTERY_PCT}."
    echo "[watchdog] Sustained VL inference has already shut this machine down once."
    echo "[watchdog] Plug in, or lower MIN_BATTERY_PCT if you accept the risk."
    exit 3
  fi
  echo "[watchdog] power: ON BATTERY at ${BATT}% -- will stop the child below ${ABORT_BATTERY_PCT}%."
  echo "[watchdog] Plugging in is strongly advised for multi-page or multi-mode runs."
fi

BASE_SWAP=$(swap_used_mb)
echo "[watchdog] nice=${NICE}  baseline swap=${BASE_SWAP}MB  limits: rss<${RSS_LIMIT_MB}MB swap<+${SWAP_GROWTH_MB}MB (once rss>${SWAP_MIN_RSS_MB}MB) free>${FREE_PCT_MIN}%"

# Constrain thread pools so the CPU isn't saturated and memory arenas stay small.
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
       VECLIB_MAXIMUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
       CPU_NUM=$THREADS FLAGS_use_mkldnn=0 TOKENIZERS_PARALLELISM=false \
       PYTORCH_ENABLE_MPS_FALLBACK=1

nice -n "$NICE" "$PY" "$@" > "$LOG" 2>&1 &
CHILD=$!
echo "[watchdog] pid=$CHILD log=$LOG"

REASON=""
PEAK=0
FREE_LOW=0
while kill -0 "$CHILD" 2>/dev/null; do
  # Sum RSS across the child and every descendant, walking the tree rather than
  # one level: pgrep -P returns direct children only, so the tesseract binary a
  # worker spawns was previously neither measured nor killed.
  PIDS=""; FRONTIER="$CHILD"
  while [ -n "$FRONTIER" ]; do
    NEXT=""
    for pp in $FRONTIER; do
      kids=$(pgrep -P "$pp" 2>/dev/null | tr '\n' ' ')
      [ -n "$kids" ] && NEXT="$NEXT $kids"
    done
    PIDS="$PIDS $NEXT"; FRONTIER="$NEXT"
  done
  RSS_KB=$(ps -o rss= -p "$CHILD" $PIDS 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ -z "$RSS_KB" ] || [ "$RSS_KB" = "0" ] && ! kill -0 "$CHILD" 2>/dev/null && break
  RSS_MB=$((RSS_KB / 1024))
  [ "$RSS_MB" -gt "$PEAK" ] && PEAK=$RSS_MB

  SW=$(swap_used_mb); SW_DELTA=$((SW - BASE_SWAP))
  FP=$(free_pct); FP=${FP:-100}
  BATT=$(battery_pct); BATT=${BATT:-100}
  SL=$(speed_limit); SL=${SL:-100}

  # Power and heat are the two ways this machine dies rather than merely slows.
  if ! on_ac && [ "$BATT" -lt "$ABORT_BATTERY_PCT" ]; then
    REASON="battery ${BATT}% < ${ABORT_BATTERY_PCT}% on battery power"
  fi
  if [ "$SL" -lt "$THERMAL_MIN" ]; then
    REASON="CPU speed limit ${SL}% < ${THERMAL_MIN}% (thermal throttling)"
  fi

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
    for pp in $PIDS; do kill -9 "$pp" 2>/dev/null; done
    kill -9 "$CHILD" 2>/dev/null
    wait "$CHILD" 2>/dev/null
    echo "[watchdog] peak RSS ${PEAK}MB. See $LOG"
    exit 99
  fi
  printf "[watchdog] rss=%sMB peak=%sMB swap+%sMB free=%s%% batt=%s%% cpu=%s%%\n" \
         "$RSS_MB" "$PEAK" "$SW_DELTA" "$FP" "$BATT" "$SL"
  sleep "$POLL"
done

wait "$CHILD"; RC=$?
echo "[watchdog] child exited rc=$RC, peak RSS ${PEAK}MB. Log: $LOG"
exit $RC
