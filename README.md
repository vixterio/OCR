# PaddleOCR-VL on an 8 GB Apple Silicon Mac

## Files

| File | What it does |
|---|---|
| `PPOCRVL_1.py` | The original script, fixed. Runs PaddleOCR-VL against a local MLX server and writes JSON + Markdown to `output/`. |
| `PPOCRVL_mlx.py` | Same thing, parameterised via `VL_SERVER_URL` / `VL_MODEL` env vars. |
| `PPOCRVL_native.py` | Runs the VL model in-process (`native` backend). **Does not fit in 8 GB** — kept for reference / a bigger machine. |
| `PPOCRVLflash.py`, `PPOCRVLtrans.py` | Empty placeholders, committed with no content. |
| `_compat.py` | Python 3.9 asyncio fix for paddlex (see below). |
| `PPOCRdet.py` | Detection only (PP-OCRv6 medium). Finds *where* text is — polygons, no characters. |
| `PPOCRrec.py` | Recognition only (PP-OCRv6 medium). Reads characters from an image already cropped to one text line. |
| `tesseract_ocr.py` | Tesseract OCR via pytesseract. No model download, no server. |
| `safe_run.sh` | Runs a script under a memory watchdog that kills it before macOS starts swap-thrashing. |
| `start_server.sh` | Starts the MLX VLM server on port 8080. |
| `requirements.txt` / `requirements-mlx.txt` | Pinned deps for the two environments. |

## Usage

### PaddleOCR-VL (layout-aware, needs the MLX server)

```bash
./start_server.sh
./safe_run.sh PPOCRVL_1.py demo.png
```

Stop the server with `kill $(cat mlx_server.pid)`.

### PP-OCRv6 detection / recognition (no server)

```bash
./safe_run.sh PPOCRdet.py demo.png   # 144 boxes on the demo page
./safe_run.sh PPOCRrec.py            # crops the first detected line, then reads it
```

Detection and recognition are two halves of one pipeline. `PPOCRdet.py` outputs
polygons; `PPOCRrec.py` expects a single-line crop and outputs characters. Give a
full page to the recogniser and you get one garbled string. To do both at once,
use the `PaddleOCR` pipeline class rather than the `TextDetection` /
`TextRecognition` model classes.

### Tesseract

```bash
brew install tesseract          # binary, English only
brew install tesseract-lang     # extra languages (chi_sim, etc.)

.venv/bin/python tesseract_ocr.py --list
.venv/bin/python tesseract_ocr.py image.png -l eng
```

Writes plain text plus a TSV of per-word boxes and confidences to
`output/tesseract/`. Tesseract needs no watchdog — it is a native binary using
tens of MB.

#### Tesseract vs PaddleOCR

Tesseract is much lighter and needs no model download, but it is weaker on dense
or non-Latin pages. On a clean English sample it scored 96.3% mean confidence
with exact text. It cannot read `demo.png` at all without `chi_sim` installed,
whereas PP-OCRv6 recognition returned a Chinese line at 0.9998 confidence.

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
