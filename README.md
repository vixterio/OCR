# PaddleOCR-VL on an 8 GB Apple Silicon Mac

## Files

| File | What it does |
|---|---|
| `PPOCRVL_1.py` | The original script, fixed. Runs PaddleOCR-VL against a local MLX server and writes JSON + Markdown to `output/`. |
| `PPOCRVL_mlx.py` | Same thing, parameterised via `VL_SERVER_URL` / `VL_MODEL` env vars. |
| `PPOCRVL_native.py` | Runs the VL model in-process (`native` backend). **Does not fit in 8 GB** — kept for reference / a bigger machine. |
| `PPOCRVLflash.py`, `PPOCRVLtrans.py` | Empty placeholders, committed with no content. |
| `_compat.py` | Python 3.9 asyncio fix for paddlex (see below). |
| `safe_run.sh` | Runs a script under a memory watchdog that kills it before macOS starts swap-thrashing. |
| `start_server.sh` | Starts the MLX VLM server on port 8080. |

## Usage

```bash
./start_server.sh
./safe_run.sh PPOCRVL_1.py demo.png
```

Stop the server with `kill $(cat mlx_server.pid)`.

## What was broken

1. **Stale venv.** `.venv` was built in `~/Desktop/leaning_coding`; 33 console scripts
   (`pip`, `paddleocr`, …) still had that path in their shebang, so they all failed with
   `bad interpreter`. Rewritten to the current path.

2. **`vllm-server` backend can't work on macOS.** vLLM has no Metal/macOS build, so nothing
   could ever listen on `:8080` and every run ended in
   `RuntimeError: Exception from the 'vlm' worker: Connection error.`

3. **The `native` backend crashed the Mac.** `paddlex/inference/utils/misc.py:33`
   only reports bf16 as available for `gpu/npu/xpu/mlu`, so on CPU the predictor falls back
   to **float32** — the 1.9 GB bf16 checkpoint expands to ~3.8 GB resident. Paddle's CPU build
   here has no bf16 *or* fp16 matmul kernel, so float32 is the only option and there is no way
   to shrink it. Against 8 GB of RAM that means swap death.

4. **`jinja2` missing** in the MLX server env — chat templating returned HTTP 500.

5. **Python 3.9 asyncio bug in paddlex.** `genai.py:527` builds an `asyncio.Semaphore` in
   `__init__` on the main thread, then awaits it on a background event loop, giving
   `RuntimeError: ... attached to a different loop`. On 3.9 a Semaphore binds to the loop at
   construction; 3.10 removed that. `_compat.py` rebuilds the semaphore inside the running loop.
   Upgrading the venv to Python 3.10+ would make the shim unnecessary.

## Why MLX

Paddle CPU float32 needs ~3.8 GB just for weights. `mlx-community/PaddleOCR-VL-4bit` is the
same model quantised to 4 bits (~0.6 GB) running on the M2 GPU.

Measured on the demo page: **peak 1.23 GB, ~250 tok/s decode** — versus the >4 GB that took
the machine down.

## The watchdog

`safe_run.sh` samples the process tree every second and SIGKILLs it if resident size, swap
growth, or free-memory percentage crosses a limit:

```bash
RSS_LIMIT_MB=3000 SWAP_GROWTH_MB=1000 FREE_PCT_MIN=8 ./safe_run.sh script.py
```

It caught the native-backend run at 2.5 GB when swap jumped 1.78 GB in one interval, and the
machine stayed responsive.
