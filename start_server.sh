#!/bin/bash
# Start the MLX VL server for a given model family.
#
#   ./start_server.sh                  # paddle, model chosen from detected RAM
#   ./start_server.sh deepseek         # DeepSeek-OCR-2
#   ./start_server.sh qwen 32          # Qwen tier for a 32 GB machine
#   ./start_server.sh --model <hf-id>  # explicit model
#
# The server holds ONE model at a time, so switching family means restarting it.
# That is deliberate: comparing two modes against different loaded models would
# produce a meaningless comparison.
set -uo pipefail
cd "$(dirname "$0")"

FAMILY="${1:-paddle}"
RAM="${2:-}"
MODEL=""
if [ "$FAMILY" = "--model" ]; then MODEL="${2:-}"; FAMILY="explicit"; fi

if [ -z "$MODEL" ]; then
  MODEL=$(.venv/bin/python - "$FAMILY" "$RAM" <<'PY'
import sys
import vl_registry as reg
fam = sys.argv[1]
ram = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else reg.detect_ram_gb()
spec = reg.SPECS.get(fam)
if spec is None:
    sys.exit(f"unknown family {fam!r}; choose from {', '.join(reg.SPECS)}")
print(spec.model_for(ram))
PY
) || { echo "$MODEL"; exit 1; }
fi

CURRENT=$(curl -sS -m 2 http://127.0.0.1:8080/v1/models 2>/dev/null \
          | .venv/bin/python -c 'import json,sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: pass' 2>/dev/null)

if [ "$CURRENT" = "$MODEL" ]; then
  echo "already serving $MODEL"; exit 0
fi
if [ -n "$CURRENT" ]; then
  echo "stopping current server (was serving $CURRENT)"
  pkill -f "mlx_vlm.server" 2>/dev/null
  sleep 2
fi

echo "starting $MODEL …"
# mlx_server_patched.py applies the DeepSeek-OCR-2 per-thread GPU stream fix and
# then delegates to mlx_vlm.server; see that file for why it is needed.
nohup .venv-mlx/bin/python mlx_server_patched.py \
  --model "$MODEL" --host 127.0.0.1 --port 8080 --max-num-seqs 1 > mlx_server.log 2>&1 &
echo $! > mlx_server.pid

until curl -sS -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; do
  kill -0 "$(cat mlx_server.pid)" 2>/dev/null || {
    echo "server failed to start:"; tail -12 mlx_server.log; exit 1; }
  sleep 2
done
echo "ready: $(curl -sS http://127.0.0.1:8080/v1/models | .venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"
