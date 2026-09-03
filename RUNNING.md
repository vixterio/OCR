# Running it on your own PDF

## One-time setup

Already done on this machine; listed so it can be reproduced elsewhere.

```bash
brew install tesseract tesseract-lang          # OCR engine + 126 language packs
/opt/anaconda3/bin/python3.12 -m venv .venv-mlx # VL server env (Apple Silicon)
.venv-mlx/bin/pip install -r requirements-mlx.txt
.venv/bin/pip install -r requirements.txt       # pipeline env
```

## Every session: start the VL server

```bash
./start_server.sh
```

It loads `mlx-community/PaddleOCR-VL-4bit` on port 8080 and is idempotent — running
it twice is harmless. Stop it with `kill $(cat mlx_server.pid)`.

## Run the hybrid pipeline on your PDF

```bash
./safe_run.sh hybrid_ocr.py /path/to/your.pdf --outdir output/mydoc
```

`safe_run.sh` is a memory watchdog; on an 8 GB machine, use it rather than calling
Python directly. For a non-English document, pass a Tesseract script model:

```bash
./safe_run.sh hybrid_ocr.py scan.pdf --lang script/Latin --outdir output/mydoc
./safe_run.sh hybrid_ocr.py scan.pdf --lang script/Cyrillic
./safe_run.sh hybrid_ocr.py scan.pdf --lang deu          # if you know the language
```

Images work too: `./safe_run.sh hybrid_ocr.py page.png`.

### What you get

| File | Contents |
|---|---|
| `output/mydoc/document.html` | the readable transcript, one section per page, every voted number hoverable for its per-engine readings |
| `output/mydoc/audit.json` | every number with all readings, confidences, dissent, margin, flags, page and bbox |

Open the HTML in a browser. Numbers needing attention are marked **⚑** and
carry a heavier border as well as colour, so the signal survives greyscale
printing and colour-blindness.

### Exit codes

- `0` — every region transcribed
- `2` — one or more regions were **not** transcribed; the HTML shows a visible
  placeholder for each. This is deliberate: an incomplete page must never be
  mistaken for a complete one. Use `--allow-gaps` to exit 0 anyway.

### Useful flags

```
--max-pages 5           process only the first 5 pages (try a long record cheaply)
--render-dpi 300        only used for PDF pages that are NOT a single embedded scan
--min-families 2        engine families needed to corroborate a separator merge
--min-margin 0.15       normalised vote margin below which a number needs review
--w-tesseract 0         drop Tesseract from the vote (e.g. CJK without chi_sim)
--vl-max-pixels N       cap what reaches the VL model; OCR engines keep full res
--allow-gaps            do not exit non-zero on untranscribed regions
```

### Scanned PDFs specifically

The pipeline prints what it found per page, e.g.

```
page 1: 2480x3508px, 300 dpi, embedded-native, scan
```

`embedded-native` means the scan was taken at its **own** resolution rather than
re-rendered — re-rendering a 150 DPI scan at 300 DPI costs four times the pixels
and adds no information. If a page reports an existing text layer, the pipeline
says so and **does not use it**: its provenance is unknown, and trusting a
stranger's OCR inside a clinical record is worse than redoing the work.

## Granite Docling

```bash
./start_server.sh granite
./safe_run.sh run_ocr.py doc.pdf --mode granite       # or granite+ocr
```

The smallest and fastest model here: 258M parameters, 0.63 GB, 12.5s per page
measured. It emits DocTags rather than prose, which `docling-core` parses into
structure deterministically. Requires `docling-core` (in requirements.txt) and the
patched server launcher, which works around a transformers bug that otherwise
demands torchvision for a model that does not need it.

## Compare against the stock pipeline

`baseline_vl.py` runs PaddleOCR-VL exactly as PaddlePaddle intends — no voting, no
pre-processing variants, no verification — and writes the pipeline's own HTML.

```bash
./safe_run.sh baseline_vl.py /path/to/your.pdf --outdir output/baseline
.venv/bin/python compare_runs.py output/baseline output/mydoc
```

The comparison prints numbers found by only one arm, and every case where the
vote overrode the VL model. Each of those is either the vote catching a VL error
or the vote introducing one — which is the question worth asking about whether
the extra machinery earns its cost.

The baseline also takes `--pipeline-version v1.6` (newer model) and `--backend`
for a non-MLX server:

```bash
./safe_run.sh baseline_vl.py doc.pdf --pipeline-version v1.6
VL_SERVER_URL=http://gpu-host:8000/v1 ./safe_run.sh baseline_vl.py doc.pdf \
    --backend vllm-server --model-name PaddlePaddle/PaddleOCR-VL
```

## Before changing the numeric layer

```bash
.venv/bin/python test_numeric.py
```

44 checks, each one a defect that was reproduced by execution — three-decimal
dose corruption, naked leading decimals, units, ranges, dates, fractions, and the
English and European fixture values. Run it before and after any edit to
`numeric.py`.

## Power: plug in before running

This is a **fanless MacBook Air M2**. Sustained VL inference is the heaviest load
it ever sees, and running it on battery has shut the machine down mid-run. The
watchdog now refuses to start on battery below 40%, stops the child below 15%,
and aborts if the CPU speed limit indicates thermal throttling:

```bash
REQUIRE_AC=1 ./safe_run.sh run_ocr.py doc.pdf --mode qwen+ocr   # refuse unless plugged in
MIN_BATTERY_PCT=60 ./safe_run.sh ...                            # stricter floor
```

`run_all_modes.sh` warns up front if you are on battery and inserts a 45s
cool-down between modes (`COOLDOWN=0` to disable), because six sequential
multi-gigabyte model loads is where the trouble happened.

Do not use `caffeinate` to keep the machine awake during these runs. Preventing
sleep while a fanless laptop is under sustained GPU load removes the last thing
that would have saved it.

## Reducing the load

Two changes cut the work substantially, and both are on by default:

- The layout predictor is built **once** per run, not once per page. Rebuilding a
  Paddle predictor per page was pure waste, and `del` does not deterministically
  free the native allocation, so it was also the likeliest trigger of the memory
  watchdog on a long document.
- Per-line VL re-reads are **off for page-granularity models** (DeepSeek, Qwen).
  Those models have already read the line as part of the whole page, so asking
  again per line cost ~10s each on DeepSeek — over 200s of GPU for a single page —
  to re-read text the OCR engines had read twice already. Measured effect on
  `deepseek+ocr`: 334s to 124s wall. Force it back on with `--vl-line-reads`.

**DeepSeek-OCR-2 is withdrawn on 8 GB.** Its two modes are commented out in
`vl_registry.py` -- uncomment both lines to reinstate on larger hardware, where
nothing else needs changing. Measured reason on 20 pages: 25 worker restarts,
pages 5 and 10 lost outright, worst content recall of the four families at
85.8%, and the machine driven to 1% free memory. `deepseek+ocr` could not
complete at all; it starved the machine until WindowServer was killed, and on
retry managed 1 page of 20 in fifty minutes. It also needs `nice 0`: it fails
with a Metal command-buffer timeout at reduced priority.

## Known limits

- **Prose is not verified.** Only numbers go through the vote. Prose comes from
  the VL model alone and is marked with a dashed underline.
- **Handwriting is never transcribed.** Regions labelled as handwriting,
  signature, seal or stamp are quarantined and shown as placeholders.
- **Ground truth now exists.** Three harnesses, below. Flag counts are still
  relative, but recall and precision are measured.
- 8 GB is tight. Peak is ~1.3 GB per page plus the VL server; the watchdog
  protects the machine but will stop a run if the system is already thrashing.


## Measuring accuracy

Three harnesses, because there are three kinds of ground truth available and
they answer different questions.

**`evaluate.py`** — the synthetic fixtures. `make_fixture.py` draws them from
literal strings, so the exact set of numbers on each page is known.

```bash
.venv/bin/python evaluate.py output/*/audit.json --table
```

**`score_bundle.py`** — the shuffled bundles. Every page carries an embedded
text layer, which is what the generator drew, so rendering and OCRing it is a
closed experiment. Adds PII-field recovery and footer recovery from the
manifests.

```bash
.venv/bin/python score_bundle.py output/b1/*/audit.json --table --rate 0.30
```

**`compare_html.py`** — the generator's own documents, against the HTML they
were rendered from. This is the strongest comparison available and the one to
prefer.

```bash
.venv/bin/python index_source_docs.py --cases 001 002 003 \
    --types hausarztbrief laborbefund --outdir work/r1
ROUND=work/r1 OUT=work/out/r1 MODES="granite paddle qwen" ./run_round.sh
.venv/bin/python compare_html.py work/out/r1 --round work/r1 --side-by-side
```

`--side-by-side` writes one openable page per document, source on the left and
the transcript on the right, which is what you want for reading the difference
rather than scoring it.

**Pair documents by patient name, never by case id.** The two corpora supplied
with this project are separate generation runs whose ids collide: 45 of 50
source cases carry a different patient under the same id, and even the 5 that
match by name have different institutions and content. Measured number overlap
between a source HTML and the corresponding bundle page was ~50%, against 100%
between that HTML and its own PDF. Grading across corpora reports a
catastrophic OCR failure that is entirely an artefact.

## Making harder inputs

Real scans are harder than born-digital renders, and the clean fixtures
saturate — seven of eleven configurations scored 100% on them.

```bash
.venv/bin/python degrade.py euro_table_sample.png --all      # resolution, JPEG, skew, speckle, blur
.venv/bin/python scanify.py merged_bundles/bundle_001.pdf --level medium
.venv/bin/python probe_cpu.py fixtures/euro_table_*.png      # CPU engines only, no GPU, seconds
```

`scanify.py` writes an image-only PDF, so `pdf_input` takes its genuine scan
path. Ground truth is unaffected: the degradations are pure image transforms of
the same render, and the scorers map a scanned copy back to the original.

## Image preprocessing

`imageprep.py` probes each page and applies only what the measurement justifies.

| | |
|---|---|
| always | probe, skew estimation, content-box crop |
| conditional | deskew rotation, background division, CLAHE, 3x3 median |
| never | any binarisation, fastNlMeansDenoising, whole-page deconvolution |

The never-list is measured. Binarisation does not erase strokes — recall stays
above 99% for Otsu, Sauvola and adaptive — it collapses precision to 27-46% by
promoting speckle into ink. `fastNlMeansDenoising` costs 1.76 s/page, 17% of
Granite's whole budget, and does not fix it.

The probe costs 79-96 ms on an A4 page at 300 dpi. `--no-imageprep` disables it,
`--no-deskew` measures skew and reports it without rotating.

## Tests

```bash
.venv/bin/python test_numeric.py   # 92 checks: separators, units, identifiers, comparators
.venv/bin/python test_vote.py      # 17 checks: family consensus, margins, unanimity
.venv/bin/python test_loops.py     # 17 checks: the three decode-loop shapes
```
