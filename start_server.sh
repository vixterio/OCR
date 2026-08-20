#!/bin/bash
# Start the MLX VLM server that PaddleOCR-VL talks to. ~0.6 GB resident.
cd "$(dirname "$0")"
if curl -sS -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "server already running on :8080"; exit 0
fi
nohup .venv-mlx/bin/python -m mlx_vlm.server \
  --model mlx-community/PaddleOCR-VL-4bit \
  --host 127.0.0.1 --port 8080 > mlx_server.log 2>&1 &
echo $! > mlx_server.pid
echo "starting server (pid $!)…"
until curl -sS -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; do
  kill -0 "$(cat mlx_server.pid)" 2>/dev/null || { echo "server failed:"; tail -5 mlx_server.log; exit 1; }
  sleep 2
done
echo "server ready on http://127.0.0.1:8080/v1"
