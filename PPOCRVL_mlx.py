"""PaddleOCR-VL driven by a local MLX server (Apple Silicon).

Why not the original vLLM setup: vLLM has no macOS/Metal build, so
`vl_rec_backend="vllm-server"` can never have a server to talk to here.
Why not `native`: Paddle's CPU build has no bf16/fp16 matmul kernels, so the
0.9B model loads in float32 (~3.8 GB) and won't fit in 8 GB of RAM.

MLX runs the same model 4-bit quantised (~0.6 GB) on the M2 GPU. Start the
server first:

    .venv-mlx/bin/python -m mlx_vlm.server \
        --model mlx-community/PaddleOCR-VL-4bit --host 127.0.0.1 --port 8080
"""
import os
import sys

import _compat

from paddleocr import PaddleOCRVL

if _compat.apply():
    print("[compat] applied Python 3.9 asyncio semaphore fix", flush=True)

SERVER_URL = os.environ.get("VL_SERVER_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.environ.get("VL_MODEL", "mlx-community/PaddleOCR-VL-4bit")

pipeline = PaddleOCRVL(
    pipeline_version="v1",
    vl_rec_backend="mlx-vlm-server",
    vl_rec_server_url=SERVER_URL,
    vl_rec_api_model_name=MODEL_NAME,
    vl_rec_max_concurrency=1,   # one page at a time: 8 GB is shared with the GPU
    use_queues=False,
)

output = pipeline.predict(sys.argv[1] if len(sys.argv) > 1 else "demo.png")

os.makedirs("output", exist_ok=True)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
print("DONE")
