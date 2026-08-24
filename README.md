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
| `hybrid_ocr.py` | All three engines combined: VL model for prose and layout, three-way confidence-weighted vote for every number. |
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

## Hybrid mode — three-engine voting on numbers

`hybrid_ocr.py` exists because the failure modes differ by content type. The VL
model is the best reader for prose and structure, but it is generative: it can
emit a plausible wrong digit with no signal that anything went wrong, and a
wrong digit in a table is worse than a wrong word in a sentence.

So prose goes to the VL model alone, and every **number** is read three times
and resolved by vote:

| Engine | Confidence source |
|---|---|
| PaddleOCR-VL | per-token logprobs from the MLX server, `exp(logprob)` |
| PP-OCRv6 medium | `rec_score` per text line |
| Tesseract | per-word confidence from its TSV output |

All three report real confidence, so the weighting is measured rather than
assumed. A candidate's weight is the sum of `engine_prior x confidence` over the
engines that produced it, which lets two moderately-confident engines outvote a
single confident outlier. Engine priors default to 1.0 and are tunable:

```bash
./safe_run.sh hybrid_ocr.py table_sample.png     # defaults
.venv/bin/python hybrid_ocr.py page.png --w-ppocr 1.2 --w-vl 0.9
```

Outputs land in `output/hybrid/`: `page.md` (VL transcription) and
`numbers.json` (every number with all three readings, their confidences, the
winning value, the margin over the runner-up, and its bbox on the page).

### Deliberate limits

- **Corrections are not silently applied to the prose.** Aligning a voted number
  back into VL markdown is ambiguous, especially in tables, so the audit trail is
  the output and the merge is left to the caller. Bad alignment corrupting good
  text is worse than a separate list.
- When engines disagree on *how many* numbers a line contains, the line is
  flagged (`token_counts_agree: false`) rather than force-aligned.
- Tesseract only sees lines that already look numeric, which keeps its
  weaknesses on dense CJK prose out of the result entirely.

### Parallelism

Layout detection runs once up front, then two lanes run concurrently on
different hardware:

- **GPU lane** — every VL call is HTTP to the MLX server process. Pure I/O in
  this process, so it overlaps with CPU work rather than fighting for the GIL.
- **CPU lane** — PP-OCRv6 detection/recognition and Tesseract.

The CPU lane hands each numeric crop to the GPU *as it finds it*, so VL
re-reads run while the CPU is still cropping later lines. Every Paddle
predictor is confined to the single CPU-lane thread, because Paddle predictors
are not thread-safe.

The run reports its own overlap:

```
wall 41.8s | CPU busy 14.8s | GPU busy 25.7s | overlapped 5.7s
```

Note the honest ceiling: the GPU is the bottleneck here (25.7s vs 14.8s), so
perfect overlap would save the CPU's time, not halve the wall clock.
