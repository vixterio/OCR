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

## Known limits

- **Prose is not verified.** Only numbers go through the vote. Prose comes from
  the VL model alone and is marked with a dashed underline.
- **Handwriting is never transcribed.** Regions labelled as handwriting,
  signature, seal or stamp are quarantined and shown as placeholders.
- **No ground-truth evaluation exists yet.** There is no measured silent-error
  rate, so treat flag counts as relative, not absolute.
- 8 GB is tight. Peak is ~1.3 GB per page plus the VL server; the watchdog
  protects the machine but will stop a run if the system is already thrashing.
