# Improvement review — scanned medical PDF to HTML

Scope: this reviews the pipeline as it stands at `b915cfb` against its actual purpose —
converting **scanned medical documents** into readable HTML, served from a **website**.

Three constraints arrived after the code was written, and each invalidates part of the
existing design rationale:

| Constraint | What it invalidates |
|---|---|
| **Scanned**, not born-digital | "Render at 200–300 DPI" — a scan's capture DPI is a ceiling. Re-rendering a 150 DPI scan at 300 adds cost and no information. |
| **Medical** | "Raise average accuracy" is the wrong objective. A silently wrong number is categorically worse than a flagged or missing one. |
| **Web service, not Mac-only** | MLX is Apple-Silicon-only by construction. The entire VL serving layer has to be replaced. |

Everything below is either **verified by execution against this repo** (marked ✅) or
explicitly labelled as an estimate. Two multi-agent adversarial reviews (13 dimensions,
170 proposals, ~3.4M tokens) produced the candidate list; the findings that survived
verification are what appear here.

---

## 0. Headline

**The numeric layer is not safe for clinical use as written.** The three-engine vote is a
sound idea and it has demonstrably caught real VL model errors, but the code around it
contains at least six defects that each produce *a wrong number reported as confident*,
which is precisely the failure mode the design exists to prevent.

None of these are exotic. All are reproducible in a few lines. They are listed first
because no accuracy, cost or throughput work matters until they are fixed.

---

## 1. Stop-ship defects (all verified by execution)

### 1.1 One correlated variant can overrule eight agreeing readings, invent a value, and report unanimity ✅

`reconcile()` at `hybrid_ocr.py:201-220` picks the reading with the **fewest tokens** among
those sharing the heaviest digit signature. I added that rule deliberately — the reasoning
was that OCR loses faint separators more often than it invents them. The rule is far more
dangerous than that reasoning allows.

Reproduced exactly:

```
9 readings in: 8 say ['12','5'], 1 (ppocr:up3x, conf 0.98) says ['12.5']
reconcile -> n=1, kept=['ppocr:up3x']
vote      -> {'value': '12.5', 'unanimous': True, 'n_engines': 1}
```

A single **correlated** reading — the same engine on a contrast-adjusted copy of the same
pixels — silently deleted a number from the page, invented one no independent engine read,
and stamped it `unanimous: True`. `12` and `5` becoming `12.5` is a ten-fold dose error.

**Fix.** A minority reading may *raise a flag*, never *decide a value*. Concretely:
require a separator-merge to be supported by at least two *independent engine families*
before it can change the token count; otherwise keep the majority tokenisation and set
`needs_review`.

**Drawbacks.** Fewer commas recovered automatically, so more numbers land in the review
queue — which is the correct direction, but it does increase human load. It also means the
`42,3` recovery I demonstrated earlier would now be flagged rather than fixed silently.

### 1.2 A missing logprob becomes confidence 1.00 ✅

`hybrid_ocr.py:301`: `math.exp(entry.get("logprob", 0.0))`. `exp(0.0)` is `1.0`, so a
backend that omits the field yields **maximum confidence for absent evidence**. This fails
open, and it becomes live the moment the backend is swapped — llama.cpp's OpenAI-compatible
endpoint does not return logprobs at all (see §2).

**Fix.** Treat a missing `logprob` as an error: fail the call, or record confidence `None`
and exclude the VL model from that vote. Probe the backend's schema at startup and refuse
to run if it does not match.

**Drawbacks.** A stricter startup contract means a backend upgrade can take the service
down rather than silently degrading it. For clinical use that is the right trade, but it
needs a clear operator error message.

### 1.3 `unanimous` does not mean unanimous ✅

`reconcile()` discards dissenting readings, then `vote()` computes `unanimous` over the
survivors only (`hybrid_ocr.py:401,406` fed from `:680,686`). Dissent is erased before it
can be recorded. Every committed audit reports `disputed: 0` partly for this reason.

**Fix.** Pass all readings into `vote()`. Compute `unanimous` over everything seen, and
keep the discarded readings in the audit as `dissenting`.

**Drawbacks.** The disputed count will jump sharply, and the HTML legend will stop looking
reassuring. That is an accurate representation of what the pipeline actually knows.

### 1.4 Provenance is keyed by value, so it can be attributed to the wrong number ✅

`annotate()` at `hybrid_ocr.py:434-441` builds `by_value` and takes `hits[0]`. When a value
repeats — routine in clinical tables — every occurrence receives the first one's readings,
confidence and disputed/suspect styling.

The root cause is that **the audit has no line identifier**. Verified: `output/hybrid/numbers.json`
holds 25 resolutions across only **3** distinct `(block, index_in_line)` keys, with `(1,0)`
occurring 23 times. `box` is the only positional key (24 distinct).

**Fix.** Add a `line_id` when a numeric line is created and key provenance on
`(line_id, index_in_line)`. Annotate during rendering, walking text nodes in order.

**Drawbacks.** Requires threading identity through the CPU lane and the renderer. Cheap,
but it touches both.

### 1.5 The number regex runs over HTML markup and corrupts attributes ✅

`hybrid_ocr.py:472,477` call `annotate(table)` on generated markup, and `annotate` applies
`NUMBER_RE.sub` to the whole string. `convert_otsl_to_html` emits `colspan="2"` for merged
cells, which becomes `colspan="<span class="num" …>2</span>"` — invalid HTML. Merged header
cells are ubiquitous in lab reports and clinical forms.

**Fix.** Parse the HTML and annotate text nodes only.

**Drawbacks.** Adds an HTML parser to the dependency set, or a hand-rolled tokeniser. The
former is safer.

### 1.6 A naked leading decimal is multiplied by ten or a thousand ✅

`NUMBER_RE` requires a leading digit, so the decimal point falls outside the match:

```
'.5 mg'  -> ['5']       '.125' -> ['125']      '.05' -> ['05']
```

Every engine reads the page correctly; **our own regex** produces the error, so no amount of
engine agreement can catch it. `.5 mg` is on error-prone-abbreviation lists precisely
because it is still written and still misread.

**Fix.** Accept a leading `.`/`,`, and flag the naked form as a prescribing hazard rather
than normalising it away.

**Drawbacks.** Slightly more false positives on version strings and ordinals.

### 1.7 Trailing-zero stripping destroys clinical precision and manufactures agreement ✅

```
'5.0' -> '5'      '10.00' -> '10'      '0.50' -> '0.5'
```

Two harms. Precision is clinical information (`0.50` and `0.5` differ in what they assert),
and canonicalising it makes different-precision readings compare equal, inflating apparent
consensus.

**Fix.** Keep the significant figures as read. Compare numerically for voting; preserve the
surface form for display and for the audit.

**Drawbacks.** Readings that differ only in trailing zeros will now be recorded as
disagreements. More review, more accurate record.

### 1.8 Digits inside analyte names enter the vote — and can fabricate a value ✅

```
'HbA1c 6.5' -> ['1', '6.5']        'CO2 24'  -> ['2', '24']
'B12 450'   -> ['12450']   <-- a number that appears nowhere on the page
```

The last one is the space-grouped-thousands rule merging `12` and `450`. Clinical text is
saturated with these tokens: HbA1c, B12, T4, CO2, O2, L5/S1, ICD-10.

**Fix.** Require a word boundary before a number, and exclude digits bound to a preceding
alphabetic token. Restrict space-grouping to contexts that already look numeric.

**Drawbacks.** Some legitimate identifiers will stop being verified. They should be
extracted as identifiers, not as quantities.

### 1.9 The unit is entirely outside the safety apparatus ✅

```
'Digoxin 125 mcg' -> ['125']
'Digoxin 125 mg'  -> ['125']    identical
```

A mcg→mg misread is a thousand-fold error with a perfect unanimous digit vote. The unit is
the multiplier, and every part of the machinery — regex, normaliser, reconcile, vote, the
suspect flag — is blind to it.

**Fix.** Capture the unit as part of the quantity, vote on `(value, unit)` as one object,
and treat µ/u/m confusion as its own escalation class.

**Drawbacks.** Substantially more parsing, and unit vocabularies vary by site and
speciality. This is the single largest piece of work in this document, and on a clinical
risk basis it is the most justified.

### 1.10 Fractions, ranges and compound doses are silently destroyed ✅

```
'1/2 tablet'  -> ['1','2']       'BP 120/80'   -> ['120','80']
'1-2 tablets' -> ['1','-2']      'INR 2.5-3.5' -> ['2.5','-3.5']   (invents a negative INR)
'2 x 500 mg'  -> ['2','500']     (the product is the dose)
```

Worse, `'1/2 tablet'` and `'1 2 tablet'` produce identical token lists, and
`suspect_lost_separator('1 2 tablet')` is `True` — so the pipeline may "recover" `1 2` into
`12` tablets.

**Fix.** Recognise these as structured expressions before tokenising, and vote on the
structure.

**Drawbacks.** Structural parsing is where most of the ambiguity lives; expect iteration.

### 1.11 `separator_recovered` measures noise, and never reaches the reader ✅

`hybrid_ocr.py:682` computes `recovered = n < max(len(...))` over **all** readings,
including the signature groups `reconcile` just rejected. Verified on
`output/hybrid_cjk/numbers.json`: **9 of 19** numbers carry the flag, and in every case the
cause is Tesseract reading Chinese as Latin garbage (`'FES 234 BI ee VARA'`) inflating the
maximum token count. The string never appears in `page.html`.

**This invalidates a number I previously reported.** When I cited "5 separators recovered"
as evidence that higher resolution helped, that figure was measuring garbage. The sound
evidence was the token count dropping 18→17 with `42.3` appearing.

**Fix.** Compute the flag only within the winning signature group, and render it.

**Drawbacks.** None. It is a bug.

### 1.12 Failed and dropped regions vanish, so an incomplete page looks complete ✅

A VL failure substitutes empty text (`:651-656`), empty blocks are skipped when assembling
the page (`:716-717`), image-class blocks are dropped from both lanes (`:582,:594`), and a
detection failure skips the block (`:599-603`). The only evidence is a line on stdout.

In a clinical context this is the worst failure mode in the repo: a transcript that reads as
complete while missing the dose that was actually prescribed.

**Fix.** Account for every block. Emit a visible placeholder for anything not transcribed,
and exit non-zero when coverage is incomplete.

**Drawbacks.** Uglier output, and some documents will refuse to produce a clean page. That
is the point.

---

## 2. Portability: replacing MLX

MLX is Apple-Silicon-only, so the serving layer must change. The client already talks raw
HTTP to an OpenAI-shaped endpoint (`vl_read`, `hybrid_ocr.py:257-304`), so the contract is
mostly portable — **with one trap**.

| Backend | Runs on | Multimodal | Per-token logprobs |
|---|---|---|---|
| **vLLM** | Linux + CUDA | yes | **yes**, on `/v1/chat/completions` |
| **llama.cpp** | CPU / CUDA / Metal / ROCm / Vulkan | yes | **only** on native `/completion` via `n_probs` |
| MLX | Apple Silicon only | yes | yes |

The trap: swapping to llama.cpp's OpenAI-compatible endpoint returns **no logprobs**, and
because of defect 1.2 every VL token would silently acquire confidence 1.00. The vote would
keep running and look healthier than before. Combined with defect 1.1, a confident VL
reading could then drive value selection unopposed.

**Recommendation.** vLLM for Linux GPU production (it is also what the original
`PPOCRVL_1.py` targeted). llama.cpp against its **native** `/completion` endpoint for
CPU-only and Mac development. Either way, add a startup conformance probe that asserts
logprobs are present and refuses to start otherwise.

**Drawbacks.** Two client code paths, or accepting vLLM's Linux/CUDA-only constraint and
losing dev/prod parity on Macs. Running the same quantisation in both environments is worth
more than squeezing prod — a different quantisation is a different model and needs its own
evaluation.

### Platform audit ✅

- `safe_run.sh` uses `sysctl vm.swapusage`, `memory_pressure` and BSD `ps` — all macOS-only.
  In containers, cgroup limits are the correct mechanism and the watchdog largely disappears.
- `PPOCRVL_native.py:14` divides `ru_maxrss` by 1024² assuming bytes. **Linux reports
  kilobytes**, so the same line reports 1024× high.
- `make_fixture.py:12` hardcodes `/System/Library/Fonts/…`.
- `_compat.py` is a Python 3.9 workaround; a 3.11+ floor deletes it.
- `requirements.txt` and `requirements-mlx.txt` exist and are pinned at the top level, but
  there is no transitive lockfile, and no CPU/CUDA split.

---

## 3. PDF input — the missing front end

There is **no PDF support**; `hybrid_ocr.py` takes one image. Verified with pypdfium2
(which ships Linux x86_64 **and** aarch64 wheels, so it is portable):

| Capability | Verified result |
|---|---|
| Distinguish scan from text layer | `get_text_range()` → **0 chars** on a pure scan |
| Native pixel dimensions | **1060×470** from image-object metadata |
| **True DPI** | **150.0** — measurable, not guessed |

This lets the pipeline read each page's real resolution and size its preprocessing and VL
pixel budget from evidence, instead of assuming a render DPI.

**Recommendations.** Refactor `main()` into `process_page(ndarray) -> PageResult` plus a
document driver; extract embedded scan images at native resolution rather than re-rendering;
per-page atomic commit with a resume manifest; account for every page and fail loudly on
gaps.

**Drawbacks.** Using an existing embedded OCR text layer as a shortcut was proposed and
**rejected** — its provenance is unknown, and trusting a stranger's OCR output in a clinical
record is worse than doing the work. It could serve as an extra voter, but only after
defect 1.3 is fixed so dissent survives.

---

## 4. Web service and efficiency

Measured: **7.75s fixed cost per invocation**, of which `import paddle` is **6.24s** and
model construction only 1.5s. On a 32s page that is ~24% of wall time discarded every run.

Measured: CPU busy 42.7s vs GPU busy 44.2s on a dense page — the lanes are **balanced**, so
one-sided speedups are Amdahl-capped at about 1.25×. And `OMP_NUM_THREADS` 2→6 gave **no
gain** (31.8s → 33.6s): the CPU lane is a sequential loop of small OCR calls, so it needs
task parallelism, not more threads.

**Recommendations.** Async job API (`202`-and-poll — never a synchronous 30–60s request);
prefork pool of **warm** workers, one page at a time per process (Paddle predictors are not
thread-safe, so this must be processes); the **page**, not the document, as the unit of
queued work; per-page deadlines with the hard rule that a failed page is never a blank page;
VL inference as a separately scaled service.

**Drawbacks.** Warm workers hold model memory permanently — roughly 1 GB each — so
concurrency is bounded by RAM, and a leak now accumulates instead of exiting. Page-level
queuing makes document-level ordering and cross-page tables harder.

---

## 5. Accuracy on scans

Highest-value, in order:

1. **Crop the detector's quadrilateral, not its bounding box.** On skewed scans the bounding
   box drags in neighbouring ink. Real gain, no resampling.
2. **Detect and correct 90/180/270 orientation before layout.** An upside-down page
   currently yields confident nonsense.
3. **Chromatic suppression variant**, so coloured stamps stop rewriting digits.
4. **De-correlate the vote.** The three preprocessing variants of one engine are counted as
   independent votes today. Worse, PP-OCRv6's `rec_score` is nearly constant (measured
   across the committed audits: min 0.905, mean 0.981–1.000, ≥0.9999 for 22–93% of
   readings), while Tesseract never reaches 1.0. Summing these as commensurable
   probabilities means the vote is dominated by the engine whose confidence carries the
   least information. One weighted vote per engine *family*, with variants as within-family
   evidence.

**On adding a fourth engine: don't, yet.** Deduplicating the correlated voters you already
have is worth more than adding another. Apple Vision was evaluated and **rejected on
portability** — it is Mac-only, so it cannot ship in a web backend. A Greek-specific Paddle
model was rejected as speculative: the Greek weakness is a *prose* problem, and the fix is
to surface the per-block VL confidence that `:302` currently computes and discards.

**Drawbacks of all preprocessing work.** If a reviewer is shown a crop as evidence, it must
be **original pixels** — otherwise the same code whose output is being checked has altered
the evidence, and a resampling artefact can create or destroy the very mark in dispute.
Keep verification crops unprocessed.

---

## 6. Cost — and where cutting is dangerous

Prompt tokens dominate ~6:1, and the VL pixel cap already landed (11,541 → 5,081 tokens,
100.1s → 31.8s at 3× render, measured).

**Worth doing:** cap generation adaptively and abort degenerate calls instead of paying for
up to 4,096 junk tokens — the 3× degeneration produced 4,276 completion tokens of repeated
markup, so this is measured, not hypothetical.

**Explicitly rejected**, because each trades safety for compute:

- *Cache VL results by perceptual hash for recurring forms* — the most dangerous proposal in
  the review. Two different patients' forms are perceptually near-identical; a hash
  collision cross-contaminates records.
- *Batch numeric-line crops into one stacked image* — converts independent reads into one
  correlated read, and one degeneration then corrupts a whole page.
- *Gate Tesseract on per-document confidence* — removes a voter exactly where the document
  is hardest.
- *Early-exit the variant loop on agreement* — the saving assumes skipped reads change
  nothing, which the separator recoveries contradict.

The honest position: this pipeline is already cheap in money terms. Spend the compute.

---

## 7. HTML, readability and the review interface

The renderer is better than I expected — `convert_otsl_to_html` produces real
`<table>/<thead>/<th>`, and every number carries its readings in a `title` attribute. What
is missing:

- **Positional keying** (defect 1.4) and **no regex over markup** (defect 1.5) — prerequisites.
- **Unverified numbers need their own visible state.** Prose numbers never enter the vote at
  all, yet render identically to voted ones.
- **Colour must not be the sole signal.** `.disputed` and `.suspect` are colour-only.
- **Embed the source image crop beside every flagged number** — the best
  readability-per-line-of-code available. It makes verification possible without opening the
  original PDF.
- **Semantic structure**: pass block labels and reading order to the renderer; `scope`,
  `caption`, `colgroup`; `lang` attributes for mixed-script content; page boundaries.
- **`page.md` leaks raw `<fcel>`/`<nl>` tokens.**
- **Do not drop headers and footers.** A proposal to honour `markdown_ignore_labels` was
  rejected: in clinical documents the header carries patient identity and the footer carries
  `Page 3 of 7` — the only available detector of a missing page.
- **Handwriting**: quarantine the region and show it as an image. Never transcribe silently,
  and never omit it.

---

## 8. Privacy and operations

Cheap and immediate: pin the VL call to loopback and bypass proxies; `os.umask(0o077)`,
per-request directories at `0700`, `O_EXCL` with mode `0600`; explicit `encoding="utf-8"` on
every write; atomic writes; a non-PHI-revealing output directory name; contain and destroy
Tesseract's temp crops.

Structural: vendor and digest-pin the model weights so containers start offline
(`HF_HUB_OFFLINE=1`) — today weights are fetched from HuggingFace at runtime, unpinned;
default-deny egress; a full run manifest per page (code version, model digests, input hash)
so an output can be tied to what produced it.

**Note on auditability.** The VL model is non-deterministic across runs even at temperature
0, so byte-identical reproduction is not achievable. Record the manifest and the readings so
the *vote* is replayable even though the generation is not.

**Drawbacks.** `numbers.json` is a PHI artefact by design — it contains recognised text. It
needs the same protection and retention policy as the source document, not a convenient
debug file left in `output/`.

---

## 9. The precondition: evaluation

The review produced 170 proposals and **not one of them can be accepted or rejected on
evidence today.** The entire justification for the voting design is five anecdotes.

Two numbers matter:

- **silent error rate** — wrong value, not flagged, per 1,000 numbers
- **review burden** — percentage of numbers flagged

Without both, a change that halves silent errors by flagging 40% of the page looks like a win
and is a worse product.

`make_fixture.py` is already most of a ground-truth generator: it renders a known table at a
chosen scale. It needs (a) emitting expected values as JSON alongside the PNG and (b) a
degradation model — blur, JPEG, skew, speckle, thin-mark dropout — so scans can be
simulated. That is PHI-free, so it can live in CI.

Then: capture reviewer corrections. A correction *is* ground truth, on real documents, at
zero labelling cost. The keys already exist in the audit.

---

## 10. The largest lever is not software

For any ongoing intake: **scan at 300 DPI, greyscale or bitonal, no auto-crop, no
scanner-side deskew, no re-JPEG.** That raises the ceiling, where everything in this
document only raises the recovery rate. It appeared in none of the 170 proposals.

---

## 11. Sequencing

1. **Defects 1.1–1.5, 1.11, 1.12** — the ones that report false confidence. Nothing else
   matters first.
2. **Evaluation harness** (§9) — otherwise steps 3+ are unfalsifiable.
3. **`process_page` refactor + PDF driver** (§3) — everything multi-page depends on it.
4. **Backend swap with a conformance probe** (§2) — before it becomes urgent.
5. **Defects 1.6–1.10** — the number-semantics work. 1.9 (units) is the largest and most
   justified.
6. **Web service + warm workers** (§4).
7. **Review UI** (§7) — arguably the actual product.
8. **Preprocessing and de-correlation** (§5).

Governance question that precedes all of it: if this output informs clinical decisions, in
the UK/EU it is plausibly a regulated medical device. That determines which of these changes
are even permissible, and it is a design input rather than a later compliance step.
