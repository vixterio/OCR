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
brew install tesseract-lang     # all 126 language packs (~686 MB)

.venv/bin/python tesseract_ocr.py --list
.venv/bin/python tesseract_ocr.py page.png --script latin   # any Latin-script language
.venv/bin/python tesseract_ocr.py page.png -l pol           # better, if the language is known
```

#### European coverage

`tesseract-lang` is one all-or-nothing bottle, so it installs every language.
The European ones present: **44 Latin** (eng fra deu spa ita por nld dan swe nor
fin isl pol ces slk slv hrv bos srp_latn hun ron lit lav est sqi mlt gle gla cym
bre cat eus glg tur ltz fao oci cos epo lat frm enm ita_old spa_old), **6
Cyrillic** (rus ukr bel bul srp mkd) and **2 Greek** (ell grc).

Prefer `--script latin|cyrillic|greek` over chaining languages. One
`script/Latin` pass reads Polish, Czech, Hungarian and Turkish without being
told which; `-l pol+ces+hun+...` is slower *and* less accurate, because each
added language widens the character set Tesseract must choose from. Verified
character-exact, diacritics included:

```
Η γρήγορη καφέ αλεπού. Σύνολο: 1.284,50 ευρώ    [script/Greek]
Быстрая коричневая лиса. Итого: 2 019,75 рублей  [script/Cyrillic]
Zażółć gęślą jaźń. Suma: 3 145,08 złotych        [script/Latin]
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

### Measured behaviour

**`table_sample.png`** (English table) — all 22 values resolved correctly across
repeated runs, no misses, no duplicates, no false separator flags.

**`euro_table_sample.png`** (Polish/Greek, European number formats) — all 12
table values correct, including `1.284,50`, `2 019,75` and `1 073,82` in the same
document.

**`demo.png`** (Chinese newspaper, no `chi_sim`) — every dispute is Tesseract
reading Chinese as Latin and emitting garbage (`2024` -> `25`, `5000` ->
`50002`). It loses every time, because its confidence lands at 0.03-0.38 while
the other two agree at 1.00. A weak engine outside its competence does not need
excluding by hand; low confidence demotes it.

#### Why the vote is worth its cost

The 4-bit VL model is not merely occasionally imprecise on digits — it fails
*hard*, and not always the same way between runs, so a single pass cannot be
trusted even at temperature 0. Errors caught so far, every one outvoted by
PP-OCRv6 and Tesseract agreeing:

| VL model read | Truth | Failure |
|---|---|---|
| `5548` @ **1.00** | `5548.39` | decimals dropped, *at full confidence* |
| `486.48` @ 0.44 | `4486.48` | leading digit dropped |
| `304.25` @ 0.54 | `3304.25` | leading digit dropped |
| `2626926.24` @ 0.24 | `26926.24` | digits duplicated |
| `111…1556.40111…` @ 0.27 | `1556.40` | degenerate repetition loop |

The first row is the important one. A confidence threshold would have accepted
it: the model was certain and wrong. Only disagreement caught it. The dropped
leading digits at 0.44-0.54 are similarly awkward — moderate confidence is not a
usable signal on its own.

This is also the honest argument for not using the VL model alone on financial
tables at 4-bit quantisation, whatever its prose quality.

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

### Measured behaviour

**`table_sample.png`** (English table, Tesseract fully capable) — all 22 table
values resolved correctly, no duplicates, no misses. One dispute, and it is the
case the whole design exists for:

```
vl=5548@1.00, ppocr=5548.39@1.00, tesseract=5548.39@0.95  ->  5548.39
```

The VL model dropped the decimals *while reporting confidence 1.00*. Two
engines outvoted it.

**`demo.png`** (Chinese newspaper, no `chi_sim` installed) — 22 numbers, 13
disputed. Every dispute is Tesseract reading Chinese as Latin and emitting
garbage (`2024` -> `25`, `5000` -> `50002`). It loses every time, because its
confidence lands at 0.03-0.38 while the other two agree at 1.00. A weak engine
outside its competence does not need to be excluded by hand; low confidence
demotes it automatically.

```
wall 50.5s | CPU busy 31.0s | GPU busy 41.8s | overlapped 26.9s
```

Overlap is much better on a real page (27s of 50s) than on the small table,
since there is more CPU work to hide behind the GPU.

If you routinely OCR CJK pages, either install `tesseract-lang` so its readings
are meaningful, or drop `--w-tesseract` to 0 to skip the noise.

### The limit of voting: shared failure modes

Agreement between engines is strong evidence, but it is not proof. If all three
make the *same* mistake, the vote reports unanimity and the audit looks clean.

This showed up on `euro_table_sample.png`. The line reads
`Wzrost 18,7% — Ανάπτυξη 42,3%`, and every engine dropped the decimal comma:

```
ppocr: 'Wzrost 18 7% — Avπτuξn 42 3%'
```

So `18,7` became two numbers, `18` and `7` — unanimously, at high confidence.
The same page's table cells (`845,09`, `987,31`) were read correctly at the same
font size, so this is specific to that mixed Polish/Greek line, not a
resolution problem; padding the crop and upscaling 4x did not recover the comma.

`hybrid_ocr.py` therefore flags it structurally rather than trusting agreement.
A digit run separated from the next by only a space, where the next run is *not*
a 3-digit group, is very likely a lost decimal separator:

```
N number(s) on lines where a decimal separator may have been lost --
unanimity here is NOT confirmation:
  [vision_footnote] ppocr read: 'Wzrost 18 7% — Avπτuξn 42 3%'
```

The check deliberately does not fire on real space-grouped thousands
(`2 019,75`, `1 000 000,00`), which is why it tests the *length* of the
following run rather than merely the presence of a space.

Every resolution also carries the raw per-engine text in `numbers.json`, so a
suspicious number can be diagnosed without re-running the page.

### Number formats

The normaliser decides which separator is decimal from evidence rather than an
assumed locale: if both `.` and `,` appear, the last one is the decimal point;
if only one appears, it groups thousands when it splits the digits into exact
threes and is decimal otherwise. So `1,284.50`, `1.284,50` and `1 284,50` all
become `1284.5`, and `42,3` becomes `42.3`.

`1,284` is genuinely ambiguous (1284 or 1.284). The three-digit-group rule
resolves it as thousands, which is the commoner intent in tabular data — but it
is a choice, not a deduction.
