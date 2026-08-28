# Six methods for turning a scanned medical PDF into HTML

This document explains what each of the six modes in `run_ocr.py` actually does, what it
costs, and what it can and cannot be trusted with. It is a design and mechanism document,
not a benchmark: there is no labelled ground-truth set for this pipeline, so no mode has a
measured accuracy figure, and none is claimed here. Every quantitative statement is marked
**measured** (reproduced by execution against this repository) or **estimated**.

The product constraint that governs everything below: the output is read by a clinician. A
number that is flagged costs a reviewer seconds. A number that is silently wrong — a dose, a
lab value — can cause harm. The two are not on the same scale, so the pipeline is not
optimised for average accuracy. It is optimised so that when it is wrong, it says so.

---

## 1. The trade-off

The six modes are a 3 × 2 grid over two independent choices.

**Axis one: what reads the page.** Three vision-language models, one per mode family.

- A *purpose-built document model* (PaddleOCR-VL 0.9B, DeepSeek-OCR-2) is trained on
  document images and emits document markup. It is small or cheap per page, it does not
  editorialise, and its output format is predictable. Its weaknesses are the weaknesses of
  its training distribution: PaddleOCR-VL is trained predominantly on Chinese and English
  and cannot read Greek prose at all (measured, §3.1).
- A *general instruction-following VLM* (Qwen3.5-VL) is not an OCR model. It handles unusual
  layouts, it follows instructions about output format, and it has by far the broadest
  language coverage. It can also paraphrase, summarise, refuse, or quietly "tidy" a number
  into a more plausible-looking one — which is the single worst failure mode available in a
  clinical document, because a tidied number looks correct.

**Axis two: whether anything checks the numbers.** Adding `+ocr` to a mode does not add a
second VL model. No mode ever loads two VL models. `run_ocr.py` checks that the
mode's model is present in the local cache before starting; it cannot verify which model is
*loaded*, because the server's `/v1/models` endpoint enumerates the HuggingFace cache rather
than the resident model, and the server hot-swaps to whatever a request names. A mismatch
therefore costs a model reload, not a silently mis-attributed comparison. What `+ocr` adds is a second and third
*independent engine family* — PP-OCRv6 and Tesseract — reading the same numeric lines, and a
confidence-weighted vote over the result.

The reason a vote exists at all is that a generative model can be confidently wrong. On the
PaddleOCR-VL stack, five distinct number errors were measured, and the important one is the
first:

| VL model read | Truth | Failure |
|---|---|---|
| `5548` at **confidence 1.00** | `5548.39` | decimals dropped, at maximum confidence |
| `486.48` at 0.44 | `4486.48` | leading digit dropped |
| `304.25` at 0.54 | `3304.25` | leading digit dropped |
| `2626926.24` at 0.24 | `26926.24` | digits duplicated |
| `111…1556.40111…` at 0.27 | `1556.40` | degenerate repetition loop |

(measured). A confidence threshold would have accepted row one: the model was certain and
wrong. Only disagreement caught it. The 0.44–0.54 rows show that moderate confidence is not
a usable signal on its own either. The model is also non-deterministic across runs at
temperature 0 (measured), so a single pass is not reproducible even in principle.

That is the case *for* the vote. The case *against* is equally measured: on a clean two-page
synthetic PDF, the vote changed nothing — baseline 14.6 s against hybrid 57.6 s for the same
40 numbers, with zero overrides (measured). Roughly a four-fold cost for no change in
output on that document. The vote is insurance against intermittent failure, not a uniform
accuracy gain. On a clean page you pay the premium and collect nothing; the question is what
a claim is worth when it lands.

---

## 2. What every mode shares

Understanding the mode differences requires knowing what is *not* different. All six modes
run the same layout pass, the same coverage accounting, the same number semantics and the
same HTML annotator (`ocr_core.py`, `numeric.py`, `hybrid_ocr.py`).

**Layout runs in every mode.** `PP-DocLayoutV2` detects blocks, sorted into reading order. In
block granularity the boxes drive the VL calls; in page granularity the VL model does not need
them, but layout still runs, because it is what identifies `handwriting`, `signature`, `seal`
and `stamp` regions for quarantine and what makes coverage accounting possible. A page model
that silently skipped a handwritten dose would be worse than a slower one that flags it.

**Handwriting is never transcribed, in any mode.** Quarantined labels render as a visible
`NOT TRANSCRIBED` placeholder with the label and the box coordinates; `figure` and `image`
render as `Figure not transcribed`.

**An incomplete page must never look complete.** A failed VL call, a reply truncated at
`max_tokens` (`finish_reason == "length"`), an empty reply and a quarantined region all produce
a visible placeholder, and the process exits non-zero unless `--allow-gaps` is passed.

**The VL backend has a contract.** `vl_read` requires per-token logprobs; a missing
`logprobs.content` array and a per-token entry missing its `logprob` are both hard errors.
Both were previously silent, and `exp(0.0) == 1.0` meant a missing logprob became *maximum
confidence for absent evidence*.

**The VL model and the OCR engines get different images from the same crop.** Measured: a 3×
rendered page made the 4-bit VL model degenerate into 10,582 characters of repeated markup
where the correct table was 377 characters, while the OCR engines read the same page correctly.
Higher resolution helps the OCR engines and breaks the 4-bit VL model, so `downscale_for_vl`
caps the pixels reaching the VL model while the CPU lane keeps full resolution. With the cap,
the 3× render cost 14% more tokens than 1× and ran no slower (measured).

**Number semantics are separate and tested.** `numeric.py` parses quantities structurally —
ranges, ratios, fractions, dates, products, units — rather than scraping digits, with
`test_numeric.py` as its regression harness (44 checks). Every rule exists because the previous
inline version corrupted a value in a way no cross-engine agreement could detect. Two that
matter below: a single `.` before exactly three digits is read as a **decimal** and flagged,
not as a thousands separator, because the thousands reading multiplied every three-decimal
dose by 1000; and units in `RISKY_UNITS` (mcg, µg, mg, g, kg, IU, U) raise a flag whenever
present, because mcg/mg is a character-level distinction a digit vote cannot see.

**Provenance is positional, never by value.** The annotator walks text nodes through an HTML
parser and consumes the number queue in reading order. Keying by value meant every repeated
value — routine in clinical tables — inherited the first occurrence's confidence and styling. If
queue and text drift out of alignment, the page carries a `Provenance alignment incomplete`
banner rather than a plausible-looking wrong tooltip.

---

## 3. The six modes

Notation: **N** = non-quarantined blocks on a page; **L** = numeric lines found on a page.

### 3.1 `paddle` — PaddleOCR-VL 0.9B alone

**Mechanism.** Layout detects N blocks. Each is cropped and sent to the VL model as its own
request, with a terse task prompt selected from a fixed table that mirrors the upstream
paddlex pipeline: `Table Recognition:` for tables, `Chart Recognition:` for charts,
`Formula Recognition:` for formulae, `OCR:` otherwise. Tables come back as OTSL
(`<fcel>`/`<ecel>` markup), which is converted to HTML by `paddlex`'s `convert_otsl_to_html`
and then annotated through the HTML parser. **N VL calls per page.**

Numbers are still extracted and recorded, but marked `verified: false`, confidence 0.0, and
rendered with a dashed underline. Without that, a no-vote mode would report zero numbers,
which would both make cross-mode comparison impossible and overstate how much the HTML is
telling you.

**Cost.** The smallest model of the three: 0.7 GB at 4-bit (measured from HuggingFace). It
is also the only family where precision has been tested directly — 4-bit against 8-bit gave
identical numbers, identical cost and Greek that was differently wrong rather than better
(measured), so quantisation is not this model's bottleneck. Against that, N calls per page
pay the per-request overhead N times, and that overhead is nearly flat in crop size:
8 × 20 px cost 211 prompt tokens and 0.99 s; 200 × 900 px cost 237 tokens and 1.24 s
(measured). Many small VL calls are dominated by their fixed cost, not their content.

**Choose it when** the document is English or Chinese, the layout is conventional, and you
want per-block failure isolation and real OTSL table structure at the lowest model footprint.

**Failure modes.**
- Every error in the §1 table is this model's. Digits dropped at confidence 1.00, leading
  digits lost, digit duplication, degenerate repetition loops, non-determinism at
  temperature 0. In this mode nothing catches any of them.
- Greek and Cyrillic prose. Measured: the title `Οικονομική έκθεση` came back as
  `Oğarkówuć`. This is the training mix, not the precision.
- High input resolution degenerates it (measured, §2). The pixel cap is load-bearing.
- Layout dependence is total: a block the layout pass does not find is never sent to the VL
  model, so it is absent from the output rather than wrong in it. Coverage accounting catches
  a *quarantined* or *failed* region; it cannot catch a region layout never proposed.
- Table rendering depends on the `paddlex` OTSL converter being importable; if it is not, the
  block renders as a visible `TABLE NOT RENDERED` gap with the raw markup.

### 3.2 `paddle+ocr` — PaddleOCR-VL, with PP-OCRv6 and Tesseract voting on numbers

**Mechanism.** Everything in §3.1, plus a CPU lane running concurrently. For each block,
PP-OCRv6 detection produces line polygons, deduplicated at IoU 0.5. Each line is read by
PP-OCRv6; **if `numeric.extract` finds no quantity in that reading, the line is skipped** and
left to the VL model as prose. Otherwise the line is padded by 6 px and read repeatedly: by
PP-OCRv6 and by Tesseract, across pre-processing variants (`raw`, `up3x` Lanczos upscale,
`otsu3x` global threshold, and `adaptive3x` local threshold on machines with headroom). A
per-line VL re-read is submitted to the GPU pool at the same time, so **N + L VL calls per
page**.

Then two stages resolve the readings.

*Reconciliation* decides how many quantities the line holds before anything votes on their
values. Readings are grouped by digit signature and the heaviest group by summed confidence
wins. Within it, the modal token count stands unless a *lower* count is corroborated by at
least two distinct engine families. That rule exists because the previous version preferred
the fewest tokens outright, and it was reproduced deleting a real number, inventing one no
independent engine had read, and stamping it `unanimous: True` at `n_engines=1` — eight
readings of `['12','5']` overruled by one correlated variant reading `['12.5']`, a ten-fold
dose error. A minority reading may now raise a flag; it may never decide a value alone.

*Voting* is confidence-weighted per position. Crucially, `family()` collapses variants of one
engine to one family, so nine readings are three families, not nine votes — the variants are
correlated looks at the same pixels. `unanimous` is computed over every value any reading
produced, including the ones reconciliation rejected, and the rejected readings are kept in
the audit as dissent.

`needs_review` is set by any of: a suspected lost separator, a `numeric.py` flag, a
reconciliation note, fewer than two supporting families, a normalised vote margin below 0.15,
or any dissenting reading at confidence 0.5 or above.

**Cost.** The dominant term is Tesseract: measured at 17.66 s of a 24.96 s CPU lane on a
32-line fixture, **71% of the lane**, at four calls per line. The measured end-to-end figure
is the baseline-versus-hybrid comparison from §1 — 14.6 s against 57.6 s on a clean two-page
PDF, roughly four-fold, with zero overrides on that document. Note that VL calls serialise
on the device (measured: 2 to 6 pool workers gave 1.01–1.04×), so the pool buys cross-lane
overlap, not intra-lane throughput.

**Choose it when** a wrong digit is consequential and the document is in a language Tesseract
has a pack for. This is the mode the whole design was built around.

**Failure modes.**
- **Prose is not verified in any mode, including this one.** Only numeric lines go through the
  vote. Prose is a single VL reading and is marked as such.
- **A line PP-OCRv6 does not read as numeric is never voted on.** The gate is
  `numeric.extract(p_text)` on PP-OCRv6's own text. If PP-OCRv6's detector misses the line, or
  reads a number so badly that no quantity parses out of it, the VL model's reading of that
  number stands unchecked — and it will not be marked unverified, because the pipeline does not
  know it missed it.
- **False unanimity.** Measured: on one line every engine dropped the same decimal comma and
  agreed. Cross-engine agreement cannot see a shared failure mode. `lost_separator_spans` is
  the structural check for exactly this — a digit run separated from a non-three-digit run by
  only whitespace — and it flags the numbers involved rather than the whole line. It is a
  heuristic, and a false negative here is a silently lost separator, the worse direction.
- **The unit is flagged, not voted.** `Digoxin 125 mcg` and `Digoxin 125 mg` differ by a
  factor of 1000 and produce an identical, perfectly unanimous digit vote. The risky-unit flag
  routes it to a human; it does not resolve it.
- **Review burden.** Every safety rule above pushes numbers into the review queue, and the
  thresholds (`--min-margin 0.15`, `--dissent-floor 0.5`, `--min-families 2`) are uncalibrated
  because there is no ground-truth set to tune them against.
- Tesseract outside its competence is safe but useless: measured, without the right language
  pack it produces garbage at 0.03–0.38 confidence and loses every vote automatically. Safe,
  but the vote is then effectively two families, and `n_families < 2` is itself a review
  trigger.

### 3.3 `deepseek` — DeepSeek-OCR-2 alone

**Mechanism.** One VL call for the whole page, prompt `Convert the document to markdown.`,
`max_tokens` 8192, and a larger pixel budget than the block models (`1280 × 28 × 28`) because
its encoder is built for dense full pages. **1 VL call per page.** Output is Markdown, so no
OTSL conversion is needed; it is rendered by a deliberately small in-house renderer handling
pipe tables, headings, lists and paragraphs. That renderer emits `{{text}}` placeholders which
are substituted afterwards, so recognised text never passes through the markup as raw
characters and the number annotator cannot corrupt a tag or an attribute. Grounding spans
(`<|ref|>`, `<|det|>`, `<|grounding|>`) are stripped if they appear.

**Cost.** 2.56 GB at 4-bit (measured from HuggingFace) — the largest of the three 4-bit
builds, and the heaviest on an 8 GB machine. Against that, it uses optical context
compression: it encodes a whole page into far fewer vision tokens than a general VLM would.
One call per page instead of N, at one call's worth of per-request overhead. Whether that
makes it cheaper per page than `paddle` on this stack is **unmeasured**.

**Operational cost, which is not a footnote.** Two properties make this the most awkward
family to deploy.

- **`AutoProcessor` cannot route its processor at all.** The MLX repository's `auto_map` names
  only `AutoConfig` and `AutoModel`, so `mlx_vlm.load` fails with
  `ValueError: Unrecognized processing class`. It needs a bespoke loader that finds mlx-vlm's
  own bundled `*Processor` class for the architecture, constructs it directly, and then
  replicates what `load_processor` would have attached — the detokenizer and stopping criteria,
  without which generation dies with
  `AttributeError: TokenizersBackend has no attribute stopping_criteria`. PaddleOCR-VL and
  Qwen3.5-VL need none of this. The obvious shortcut, `trust_remote_code=True`, is **wrong
  rather than a fix**: it makes transformers fetch the repository's PyTorch implementation and
  then demand `addict`, `matplotlib`, `torch` and `torchvision`. Both model and processor load
  with `trust_remote_code=False`, so **no code from the model repository is executed** — passed
  as an explicit `False`, because omitting it makes transformers prompt on stdin and hang a
  non-interactive worker forever. The supply-chain exposure that would otherwise sit here is
  closed; pinning the revision and vendoring the weights remains sound PHI practice but is no
  longer load-bearing.
- It **cannot run under `mlx_vlm.server` at all.** `get_input_embeddings` calls `.item()`,
  which synchronises, and the server generates on an `asyncio.to_thread` worker whose thread
  does not own the stream the input arrays were created on:
  `RuntimeError: There is no Stream(gpu, N) in current thread`. Verified in mlx-vlm 0.6.15
  and 0.6.17, and verified *not* to be a model defect — the same model produces correct output
  under `mlx_vlm.generate`. It therefore runs through a persistent single-threaded worker
  (`vl_worker.py`, one JSON request per line on stdin) rather than the HTTP server the other
  two families use. That worker is fragile in a specific way worth knowing about: mlx-vlm and
  the bundled processor print progress to stdout, which would corrupt a line-delimited JSON
  protocol, so stdout is redirected to stderr across both load and generation and every
  protocol message is written to `sys.__stdout__` directly.

**Choose it when** you want one call per page on a document-trained model, the pages are dense
or multi-column, and the bespoke serving path — single-threaded worker, hand-built processor —
is acceptable in your deployment.

**Failure modes.**
- **One call is one point of failure for the whole page.** A reply truncated at `max_tokens`
  marks every block on the page truncated; a contract violation marks every block failed. In
  block granularity the same fault costs you one block. The blast radius is the page.
- No per-block confidence and no per-block status: the page is `ok` or it is not.
- **Reading-order drift.** The annotator consumes numbers positionally. If the model's
  Markdown reading order differs from the layout's top-to-bottom line order, provenance
  drifts; the count mismatch is detected and banner-flagged, but a same-count reordering
  would not be.
- Optical compression is lossy by construction. Whether it drops faint marks — a thin decimal
  comma — more or less often than an uncompressed encoder is **unmeasured here**.
- Nothing verifies any number in this mode.

### 3.4 `deepseek+ocr`

**Mechanism.** §3.3 plus the CPU lane and vote from §3.2, unchanged. Note the arithmetic:
the per-line VL re-read uses this family's `line_prompt` (`Free OCR.`), so the call count is
**1 + L per page**. Once verification is on, the page model's one-call advantage applies only
to the transcription pass; the L verification calls are the same in both granularities. On a
page with many numeric lines, `1 + L` and `N + L` converge.

**Choose it when** you want DeepSeek's page-level layout handling *and* a check on the
numbers. It is the natural mode for a dense multi-column lab report.

**Failure modes.** Every one in §3.2 (unverified prose, the PP-OCRv6 numeric gate, false
unanimity, units, review burden), plus every one in §3.3 (page-wide blast radius, reading-order
drift, the deployment constraints). The vote does mitigate one of them usefully: when the VL
model collapses, the numbers still come from the OCR engines. That was observed on the
PaddleOCR-VL stack, where the numbers stayed correct through a total degeneration of one
voter (measured) — the same structural property applies here, though it has not been
reproduced on DeepSeek specifically.

### 3.5 `qwen` — Qwen3.5-VL alone

**Mechanism.** One VL call per page, like §3.3, but the prompt has to do much more work
because the model is not an OCR model. It explicitly instructs: transcribe into Markdown,
preserve reading order, represent every table as a Markdown table, transcribe numbers exactly
as printed including decimal separators, do not summarise or explain, output only the
transcription. `strip_patterns` then removes the chatter that arrives anyway — leading
`Here's…:` lines and stray code fences. **1 VL call per page.**

**Cost.** The size ladder is chosen from detected RAM: 2B at 1.72 GB and 4B at 3.03 GB
(both 4-bit, measured from HuggingFace), then Qwen3.5-9B, then Qwen3.8-27B. Two facts
constrain the top of that ladder: Qwen3.8-27B is vision-capable (architecture `qwen3_5`) but
needs roughly 48 GB, and the `keXjos/Qwen3.8-9B-mlx-4Bit` build has **no `vision_config` at
all** — it is a text-only build and unusable for OCR. So the usable ladder is
2B → 4B → 9B (Qwen3.5) → 27B (Qwen3.8). No wall-clock or token cost has been measured for
this family on this stack.

**Choose it when** the document defeats the other two: an unusual layout, or a language
outside PaddleOCR-VL's Chinese/English training mix. Greek and Cyrillic are the concrete
cases. It is also the only family where you can steer the output format by editing a prompt
rather than post-processing markup.

**Failure modes.** These are qualitatively different from an OCR model's, and worse for this
product.

- **It can paraphrase or summarise.** An OCR model that fails produces garbage you can see. A
  VLM that summarises produces fluent, plausible, shorter text. Nothing downstream detects
  that the output is a summary rather than a transcription.
- **It can "tidy" a number.** Reformatting `1.284,50` into `1,284.50`, or normalising `0.50`
  to `0.5`, changes the surface form the numeric layer parses. Precision is clinical
  information; `numeric.py` preserves trailing zeros deliberately, and a model that removes
  them upstream defeats that.
- **It can refuse.** A refusal produces an empty or off-topic reply. Empty is handled — the
  block is marked `empty` with a visible placeholder. An off-topic reply is not detectable by
  the pipeline.
- **`strip_patterns` is a text edit applied after generation.** It removes chatter; it can
  also remove a line of real content that happens to match. The patterns are anchored and
  narrow for that reason, but the risk is structural.
- Nothing verifies any number in this mode, and there is no measured error inventory for this
  family at all — the §1 table is PaddleOCR-VL's, not Qwen's.

### 3.6 `qwen+ocr`

**Mechanism.** §3.5 plus the CPU lane and vote. **1 + L VL calls per page.**

This is the mode where the vote's value is easiest to argue on mechanism and hardest to
quantify. The specific thing a general VLM does wrong — silently regularising a number — is
exactly what a cross-engine vote is positioned to catch, because PP-OCRv6 and Tesseract have
no notion of what a number *ought* to look like. Against that, the vote covers numbers only.
A paraphrased sentence or a summarised paragraph passes through untouched and unmarked, and
that is a large uncovered surface for a general VLM.

**Choose it when** you need broad language coverage *and* a check on the numbers. Install the
right Tesseract language pack, or the third family contributes nothing but low-confidence
noise.

**Failure modes.** Every one in §3.2 and every one in §3.5. The prose exposure in §3.5 is the
one the vote does not reduce.

---

## 4. The three comparisons that matter

### Page granularity against block granularity

| | Block (`paddle`) | Page (`deepseek`, `qwen`) |
|---|---|---|
| VL calls, no verification | N per page | 1 per page |
| VL calls, with verification | N + L | 1 + L |
| Needs a layout pass to work | Yes — it supplies the crops | No |
| Layout runs anyway | Yes | Yes — for quarantine and coverage |
| Per-request overhead paid | N times | Once |
| Blast radius of one bad reply | One block | The whole page |
| Per-region confidence and status | Yes | No |
| Table output | OTSL, converted | Markdown, rendered directly |

The per-request overhead is the crux, and it is nearly flat in crop size (measured, §3.1):
8 × 20 px cost 211 prompt tokens, 200 × 900 px cost 237. Block granularity therefore pays
roughly N times a fixed floor to buy per-region failure isolation and per-region confidence.
Page granularity pays it once and gives both up. Which is cheaper in wall-clock terms on real
scanned pages is **unmeasured** — it depends on N, and N depends on the document.

Note also that verification erases most of the difference. `N + L` against `1 + L` converges
as L grows, and a numeric-dense clinical table is precisely the case where L is large.

### Purpose-built OCR model against general VLM

The purpose-built models fail *loudly and locally*: garbage markup, a repetition loop, a
dropped digit. The general VLM fails *quietly and globally*: a fluent paraphrase, a tidied
number, a summarised paragraph. Loud local failure is far easier to build machinery around —
which is what the placeholders, the coverage accounting and the non-zero exit code are.

The compensating advantage is coverage. PaddleOCR-VL's Greek failure is measured and is not a
precision problem (4-bit and 8-bit were equally wrong, differently), so no amount of hardware
fixes it. For a Greek or Cyrillic document the general VLM is the only one of the three with
a plausible claim, and that claim is unmeasured here.

### Verified against unverified

"Verified" in this pipeline means one narrow thing: **numbers on lines that PP-OCRv6 read as
containing a number were read by three engine families and resolved by a confidence-weighted
vote.** It does not mean the prose was checked, the units were resolved, the layout was
complete, or the handwriting was read.

An unverified mode is not merely less accurate — it is *differently honest*. It renders every
number with a dashed underline and states in the document footer that all numbers are single
readings from one model with no cross-check. That is a truthful HTML page. A `+ocr` mode's
page is also truthful, and additionally flags the numbers where the engines disagreed.

The one measured comparison of the two is the 14.6 s / 57.6 s zero-override result (§1). Read
it as: on a clean document the vote is pure overhead; the argument for it rests entirely on
the intermittent hard failures in the §1 table, which a single pass cannot detect and a
confidence threshold does not catch.

---

## 5. The six at a glance

N = non-quarantined blocks per page; L = numeric lines per page. All model sizes are 4-bit
MLX builds, measured from HuggingFace. "Expected cost" is structural and estimated except
where a measured figure is named.

| Mode | VL calls / page | Model size | Expected cost | What is verified | Main risk |
|---|---|---|---|---|---|
| `paddle` | N | 0.7 GB | Lowest footprint; N × a per-request floor that is flat in crop size (measured) | Nothing. Every number is one reading | The §1 error inventory, unchecked — including a dropped decimal at confidence 1.00 |
| `paddle+ocr` | N + L | 0.7 GB | Highest CPU cost; Tesseract is 71% of the CPU lane (measured); 14.6 s → 57.6 s on a clean 2-page PDF (measured) | Numbers on PP-OCRv6-numeric lines, by three families | Review burden, and false unanimity when all engines share a failure |
| `deepseek` | 1 | 2.56 GB | One call per page via optical compression; largest 4-bit footprint | Nothing | One bad reply loses the whole page; no HTTP server path, so a bespoke worker and loader |
| `deepseek+ocr` | 1 + L | 2.56 GB | 1 + L converges on N + L as L grows | Numbers on PP-OCRv6-numeric lines, by three families | As above, plus the §3.2 gaps; unverified prose |
| `qwen` | 1 | 1.72 GB (2B) / 3.03 GB (4B) | One call per page; unmeasured on this stack | Nothing | Silent paraphrase, tidied numbers, refusal — failures that look like success |
| `qwen+ocr` | 1 + L | 1.72 / 3.03 GB | Unmeasured on this stack | Numbers on PP-OCRv6-numeric lines, by three families | Prose paraphrase is entirely uncovered by the vote |

Two model-availability facts that constrain the Qwen column: Qwen3.8-27B is vision-capable
(architecture `qwen3_5`) but needs roughly 48 GB, and `keXjos/Qwen3.8-9B-mlx-4Bit` has no
`vision_config` — it is a text-only build and cannot do OCR at all.

---

## 6. Which would I choose

These are mechanism arguments, not measured rankings. Where a measurement supports a choice
it is named; where none exists, that is said.

**An English clinical table** — `paddle+ocr`. Block granularity gives per-block failure
isolation and genuine OTSL table structure, the model footprint is the smallest of the three,
and Tesseract is fully competent in English so the third voter is real rather than
noise-at-low-confidence. Supporting measurement: on an English table fixture all 22 table
values resolved correctly, and the single dispute was the VL model dropping decimals at
confidence 1.00 while PP-OCRv6 and Tesseract agreed — the exact case the design exists for.

**A Greek or Cyrillic document** — `qwen+ocr`, with the Tesseract pack installed
(`--lang script/Greek` or `script/Cyrillic`). `paddle` is disqualified by measurement:
`Οικονομική έκθεση` came back as `Oğarkówuć`, and 4-bit against 8-bit was differently wrong
rather than better, so the fault is the training mix. Qwen has the broadest language coverage
of the three; that this translates into better Greek transcription is **unmeasured**. Without
the language pack, Tesseract's output is garbage at 0.03–0.38 confidence — it loses every vote
automatically, which is safe, but the vote is then two families and `n_families < 2` flags
every number for review. If you cannot install the pack, `qwen` alone is the honest choice:
you get an unverified page that says it is unverified, rather than a two-family vote pretending
to be three.

**A handwriting-heavy scan** — the mode barely matters, because **no mode transcribes
handwriting**. `handwriting`, `signature`, `seal` and `stamp` regions are quarantined in all
six, rendered as visible placeholders, and the run exits 2 unless `--allow-gaps`. So choose
for the printed remainder — a `+ocr` mode — and plan for a human pass over the quarantined
regions. The risk to watch is not transcription quality but *layout recall*: a handwritten
annotation the layout model does not label as handwriting will be fed to the VL model like any
other text and guessed at. If you want handwriting read, that is a decision to change the
quarantine list, not a mode choice — and this pipeline's position is that a guessed handwritten
dose is worse than a flagged gap.

**A high-volume cheap pass** — a page-granularity mode without verification, so `deepseek` or
`qwen`. `deepseek` is the stronger structural candidate: it is document-trained and its
optical compression means one page becomes far fewer vision tokens than a general VLM would
produce. The counterweights are operational rather than about quality: it cannot run under
`mlx_vlm.server` at all, so it needs the single-threaded worker and the bespoke processor
loader, and that is the most bespoke serving path of the three. If that is unacceptable in
your deployment, `paddle` has the smallest model (0.7 GB) but its cost
scales with block count. **Which is actually cheapest per page on this stack is unmeasured**,
and it will depend on N for your documents. Measure it on your own corpus before committing.

**Safety-critical dose extraction** — a `+ocr` mode, and the honest answer is that the mode
choice is the smallest part of it. The vote catches the failure in the §1 table; it does not
make dose extraction safe, for four reasons that are all in the code:

1. The unit is flagged, not voted. `125 mcg` and `125 mg` give an identical unanimous digit
   vote and differ by a factor of 1000.
2. Every engine can share a failure. One measured line had all three drop the same decimal
   comma and agree.
3. A line PP-OCRv6 does not read as numeric never enters the vote, and is not marked
   unverified.
4. Prose is never verified in any mode, and a dose can appear in prose.

So: run `paddle+ocr` for English or `qwen+ocr` for other scripts, do **not** pass
`--allow-gaps`, work the review queue rather than counting it, retain `audit.json` as the PHI
artefact it is, and treat the HTML banner literally — it is an unreviewed machine
transcription, not a verified record.

---

## 6b. Three operational traps found while making these run

These cost real time and are not obvious from any documentation, so they are recorded here.

**DeepSeek-OCR-2 cannot run under `mlx_vlm.server` at all.** Its
`get_input_embeddings` calls `.item()`, which synchronises, while the server generates on an
`asyncio.to_thread` worker whose thread does not own the stream the input arrays were created
on:

```
RuntimeError: There is no Stream(gpu, 2) in current thread.
    at mlx_vlm/models/deepseekocr_2/deepseekocr_2.py:86
```

Present in mlx-vlm 0.6.15 and 0.6.17 (measured). It is not a model defect — the same model
and input succeed under `mlx_vlm.generate`. Wrapping the call in a stream context does not
help, because the arrays are already bound to another thread's stream. Hence the persistent
worker in `vl_client.py`, which generates on its main thread.

**`AutoProcessor` cannot route DeepSeek-OCR-2, and `trust_remote_code=True` is a trap, not
the fix.** The MLX repository's `auto_map` names only AutoConfig and AutoModel, so
transformers raises `ValueError: Unrecognized processing class`. Setting
`trust_remote_code=True` then makes it fetch the repository's *PyTorch* implementation, which
demands torch, torchvision, addict and matplotlib. mlx-vlm ships its own processor, and it
loads with `trust_remote_code=False` — so **no code from the model repository is executed**,
which removes what would otherwise be a supply-chain concern for a PHI pipeline (measured).

Assembling model and processor by hand instead produced a Metal command-buffer timeout:
`mlx_vlm.load` does more than build two objects. `vl_worker.py` therefore substitutes only
the failing processor step inside mlx-vlm's own loader.

**Run GPU-bound modes at normal priority.** Metal command buffers have an execution deadline,
and a niced process cannot always feed the GPU fast enough to meet it. Under `nice 10`
DeepSeek-OCR-2 failed with `[METAL] Command buffer execution failed: GPU Timeout` on input it
processed correctly in 20.5s at `nice 0` (measured). `safe_run.sh` now defaults to `NICE=0`.

A consequence worth stating for the comparison: DeepSeek's `GenerationResult.logprobs` comes
back empty on this path, so there is **no per-token VL confidence** in the DeepSeek modes.
Rather than let a missing signal become a high one, that reading votes with a declared prior,
does not count towards the two-family corroboration rule, and forces review. The audit
records `vl_confidence_measured: false`. This is a real asymmetry between the DeepSeek modes
and the other four, and it should be taken into account when comparing them.

## 7. What cannot be concluded without a labelled ground-truth set

There is no ground-truth set for this pipeline. That is not a gap in the documentation; it is
the reason several natural questions have no answer here, and the reason no mode is described
above as most accurate.

**Not concludable today:**

- **Any accuracy ranking of the six modes.** Not "paddle+ocr beats paddle", not
  "deepseek beats qwen", not "the vote improves accuracy". None of it has been measured.
- **The silent error rate** — a wrong value that was *not* flagged, per 1,000 numbers. This is
  the only number that matters for the clinical claim, and it is unknown for every mode.
- **The review burden** — what percentage of numbers each mode flags, and what fraction of
  those flags are genuine. Without it there is no way to say whether a mode is usable at
  volume.
- **Whether the review thresholds are set correctly.** `--min-margin 0.15`,
  `--dissent-floor 0.5` and `--min-families 2` were chosen by reasoning, not by calibration.
  They trade silent errors against review load, and the exchange rate is unmeasured.
- **Whether the vote's overrides are net-positive.** Each override is either the vote catching
  a VL error or the vote introducing one. `compare_runs.py` lists them; only a labelled set
  can classify them.
- **Anything about DeepSeek-OCR-2 or Qwen3.5-VL transcription quality.** Every accuracy
  measurement in this repository is on the PaddleOCR-VL stack. The §1 error table is
  PaddleOCR-VL's alone; it is evidence about that model, not about generative OCR in general.

**Two specific traps in the evidence that does exist:**

*Selection bias in the error inventory.* The five caught errors were found *because* the vote
caught them. They are therefore evidence that the vote catches some errors, and no evidence
whatsoever about the errors it misses. The misses are, by construction, the ones that look
unanimous.

*Agreement is not truth.* `compare_modes.py` can report cost exactly and agreement exactly,
and it says so in its own docstring: treat the agreement column as a lead to investigate, not
as a score. A number every method reads identically is probably right; a faint decimal
separator that every engine loses also looks unanimous. Cross-engine agreement is blind to
shared failure modes by definition, and shared failure modes are the ones that reach a
clinician.

**What would settle it.** A labelled set of representative scanned pages — including the ugly
ones: fax, photocopy, skew, stamps over text, mixed-locale numbers — and two figures per mode:
silent error rate, and review burden. Until those exist, the defensible claims about this
pipeline are all mechanical: what it checks, what it does not check, and where it says so on
the page.
