"""Hybrid OCR: PaddleOCR-VL for layout and prose, three-engine voting for numbers.

Rationale
---------
The VL model is the best reader for prose and structure, but a generative model
can silently produce a plausible wrong digit, and a wrong digit in a table is
worse than a wrong word in a sentence. So numbers get read three times --
PaddleOCR-VL, PP-OCRv6 recognition, and Tesseract -- and each numeric token is
resolved by confidence-weighted vote. Prose is left to the VL model alone.

All three engines report real confidence:
  * VL model   -- per-token logprobs from the MLX server (exp(logprob))
  * PP-OCRv6   -- rec_score per text line
  * Tesseract  -- per-word confidence from its TSV output

Parallelism
-----------
Two lanes run concurrently against different hardware:

  GPU lane -- every VL call is an HTTP request served by the MLX process on the
              Metal GPU. Issued from a thread pool; the requests are pure I/O
              here, so they overlap with CPU work rather than competing for the
              GIL.
  CPU lane -- PP-OCRv6 detection/recognition (Paddle, CPU) and Tesseract.

Layout detection runs once up front because both lanes need the blocks. After
that the lanes are independent: the GPU transcribes whole blocks while the CPU
crops and re-reads numeric lines. Every Paddle predictor is confined to the
single CPU-lane thread -- Paddle predictors are not thread-safe, and this keeps
exactly one of them live at a time.

Usage
-----
    ./start_server.sh
    ./safe_run.sh hybrid_ocr.py table_sample.png
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SERVER_URL = os.environ.get("VL_SERVER_URL", "http://127.0.0.1:8080/v1")
VL_MODEL = os.environ.get("VL_MODEL", "mlx-community/PaddleOCR-VL-4bit")

# Layout labels the VL model is not asked to transcribe.
IMAGE_LABELS = {"image", "figure", "chart_image", "header_image", "footer_image"}

# Prompts mirror paddlex/inference/pipelines/paddleocr_vl/pipeline.py:308-330 so
# direct calls behave like the packaged pipeline.
def prompt_for(label: str) -> str:
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    return "OCR:"


# A number, allowing thousands separators, decimals, sign, percent and currency.
# Two alternatives, longest-first:
#   1. space/apostrophe-grouped thousands ("2 019,75", "1'000'000") -- the
#      grouping separator is only accepted when it really splits digits into
#      threes, so "in 2024 3 people" reads as two numbers, not "20243".
#   2. plain runs with dot and/or comma separators ("1,284.50", "42,3", "5548").
NUMBER_RE = re.compile(
    r"[-+\u2212]?[$\u20ac\u00a3\u00a5]?\d{1,3}(?:[\u00a0\u202f\u2007 '\u2019]\d{3})+(?:[.,]\d+)?"
    r"|[-+\u2212]?[$\u20ac\u00a3\u00a5]?\d+(?:[.,]\d+)*"
)

_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
_timeline: list[tuple[str, str, float, float]] = []
_timeline_lock = threading.Lock()


def record(lane: str, what: str, t0: float, t1: float) -> None:
    with _timeline_lock:
        _timeline.append((lane, what, t0, t1))


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def normalise_number(raw: str) -> str | None:
    """Canonical form so the same value written differently compares equal.

    Handles both conventions, which matters as soon as the document is European:
    '1,284.50', '1.284,50' and '1 284,50' are all 1284.5.

    Which separator is the decimal point is decided from evidence rather than an
    assumed locale:
      * both '.' and ',' present -> whichever comes last is the decimal point
      * only one present         -> it groups thousands if it splits the digits
                                    into exact threes, otherwise it is decimal

    '1,284' is genuinely ambiguous (1284 or 1.284); the three-digit-group rule
    resolves it as thousands, which is the commoner intent in tabular data.
    Returns None for anything that isn't really a number, so junk never votes.
    """
    s = unicodedata.normalize("NFKC", raw).strip()
    s = s.replace("\u2212", "-")
    # Space-ish and apostrophe separators carry no other meaning inside a number.
    for ch in ("\u00a0", "\u202f", "\u2007", " ", "'", "\u2019"):
        s = s.replace(ch, "")
    s = re.sub(r"[$\u20ac\u00a3\u00a5]", "", s)
    sign = "-" if s.startswith("-") else ""
    s = s.lstrip("+-")

    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thousands = "," if dec == "." else "."
        s = s.replace(thousands, "").replace(dec, ".")
    elif has_comma:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif has_dot and re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")

    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return sign + s


def dedupe_boxes(boxes, iou_thresh: float = 0.5):
    """Drop near-duplicate line boxes.

    Detection sometimes returns two boxes over the same line, which would
    otherwise resolve the same number twice and inflate the audit.
    """
    kept = []
    for box in boxes:
        ax1, ay1, ax2, ay2 = box
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        dup = False
        for bx1, by1, bx2, by2 in kept:
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            if inter / float(area_a + area_b - inter) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


# A digit run separated from the next by only a space, where the following run
# is not a 3-digit group, is very likely a decimal separator the OCR lost:
# "18,7%" misread as "18 7%". Every engine can drop the same faint comma, so the
# vote sees unanimity and reports false confidence -- this is the one failure
# mode agreement cannot catch, hence the explicit check.
LOST_SEP_RE = re.compile(r"\d[\u00a0\u202f ]+(\d+)")


def suspect_lost_separator(text: str) -> bool:
    for m in LOST_SEP_RE.finditer(text or ""):
        if len(m.group(1)) != 3:
            return True
    return False


def preprocess_variants(img):
    """Yield (name, image) readings of the same crop.

    Measured on a line where the decimal comma was lost: padding alone recovered
    one comma, upscale+Otsu recovered a different one, and neither recovered
    both. Since the variants fail independently, running several and letting them
    vote recovers more than any single choice of pre-processing.
    """
    yield "raw", img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    yield "up3x", cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    # Otsu turns a faint sub-pixel comma into solid black, which is exactly the
    # mark thin-glyph loss destroys.
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu3x", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)


def digit_signature(tokens) -> str:
    """The digits alone, ignoring where separators fell."""
    return "".join(re.sub(r"\D", "", t) for t in tokens)


def reconcile(readings):
    """Agree on how many numbers a line holds before voting on their values.

    readings: [(source, tokens, conf)].

    Groups readings by digit signature and keeps the heaviest group, then -- and
    this is the point -- within that group prefers the reading with the *fewest*
    tokens. Same digits in fewer tokens means more separators were recovered,
    and OCR loses faint separators far more often than it invents them. That
    asymmetry is what turns ['18', '7'] back into ['18.7'].
    """
    if not readings:
        return 0, []
    weight_by_sig = defaultdict(float)
    for _, toks, conf in readings:
        weight_by_sig[digit_signature(toks)] += conf
    best_sig = max(weight_by_sig.items(), key=lambda kv: kv[1])[0]
    same = [r for r in readings if digit_signature(r[1]) == best_sig]
    n = min(len(r[1]) for r in same)
    return n, [r for r in same if len(r[1]) == n]


def numeric_tokens(text: str) -> list[str]:
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        n = normalise_number(m.group())
        if n is not None:
            out.append(n)
    return out


# --------------------------------------------------------------------------- #
# GPU lane: VL model over HTTP
# --------------------------------------------------------------------------- #
VL_MAX_PIXELS = 1024 * 28 * 28   # overridden from --vl-max-pixels


def downscale_for_vl(img):
    """Cap the pixels sent to the VL model.

    Measured: feeding a 3x-rendered page straight to the 4-bit model made it
    degenerate -- it produced 10,582 characters of repeated junk in place of a
    377-character table, while the OCR engines read the same page correctly. The
    mlx-vlm-server backend ignores paddlex's max_pixels, so the cap has to be
    applied here. High resolution helps the OCR engines and hurts the VL model,
    so the two get different images from the same crop.
    """
    h, w = img.shape[:2]
    n = h * w
    if n <= VL_MAX_PIXELS or n == 0:
        return img
    scale = (VL_MAX_PIXELS / n) ** 0.5
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def vl_read(image_bgr, label: str = "text", max_tokens: int = 4096, timeout: int = 300):
    """Transcribe one crop with the VL model. Returns (text, mean_conf, tokens).

    tokens is [(token_text, confidence)], derived from per-token logprobs, which
    is what lets a single digit inside a long string carry its own confidence.
    """
    import urllib.request

    ok, buf = cv2.imencode(".png", downscale_for_vl(image_bgr))
    if not ok:
        return "", 0.0, []
    b64 = base64.b64encode(buf.tobytes()).decode()
    payload = {
        "model": VL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt_for(label)},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
    }
    req = urllib.request.Request(
        f"{SERVER_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    usage = data.get("usage") or {}
    with _timeline_lock:
        _usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        _usage["completion_tokens"] += usage.get("completion_tokens", 0)
        _usage["calls"] += 1
    choice = data["choices"][0]
    text = choice["message"]["content"] or ""
    toks = []
    for entry in (choice.get("logprobs") or {}).get("content") or []:
        toks.append((entry.get("token", ""), math.exp(entry.get("logprob", 0.0))))
    conf = sum(c for _, c in toks) / len(toks) if toks else 0.0
    record("GPU", f"vl:{label}", t0, time.time())
    return text, conf, toks


def vl_token_confidences(tokens, wanted: list[str]) -> dict[str, float]:
    """Map each numeric value to the confidence of the tokens that spelled it.

    The model emits digits in arbitrary chunks ('1', ',28', '4.50'), so tokens
    are walked and accumulated into numbers rather than matched one-to-one.
    """
    joined = "".join(t for t, _ in tokens)
    confs: dict[str, list[float]] = defaultdict(list)
    pos = 0
    spans = []
    for tok, c in tokens:
        spans.append((pos, pos + len(tok), c))
        pos += len(tok)
    for m in NUMBER_RE.finditer(joined):
        val = normalise_number(m.group())
        if val is None:
            continue
        overlap = [c for a, b, c in spans if b > m.start() and a < m.end()]
        if overlap:
            confs[val].append(min(overlap))  # weakest token governs the number
    return {k: max(v) for k, v in confs.items() if k in set(wanted)} if wanted else {
        k: max(v) for k, v in confs.items()
    }


# --------------------------------------------------------------------------- #
# CPU lane: PP-OCRv6 + Tesseract
# --------------------------------------------------------------------------- #
class CpuEngines:
    """Paddle + Tesseract, all confined to one thread."""

    def __init__(self):
        from paddleocr import TextDetection, TextRecognition

        self.det = TextDetection(model_name="PP-OCRv6_medium_det")
        self.rec = TextRecognition(model_name="PP-OCRv6_medium_rec")
        import pytesseract

        self.pt = pytesseract

    def lines(self, crop) -> list[np.ndarray]:
        t0 = time.time()
        res = next(iter(self.det.predict(crop)))
        record("CPU", "ppocr:det", t0, time.time())
        return list(res["dt_polys"])

    def rec_line(self, line_img):
        t0 = time.time()
        res = next(iter(self.rec.predict(line_img)))
        record("CPU", "ppocr:rec", t0, time.time())
        return res.get("rec_text", ""), float(res.get("rec_score", 0.0))

    def tess_line(self, line_img, lang: str):
        """Tesseract on one line crop -> (text, per-word confidences)."""
        t0 = time.time()
        from PIL import Image

        rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        tsv = self.pt.image_to_data(Image.fromarray(rgb), lang=lang, config="--psm 7")
        words, confs = [], []
        for row in tsv.splitlines()[1:]:
            parts = row.split("\t")
            if len(parts) == 12 and parts[11].strip():
                words.append(parts[11])
                try:
                    c = float(parts[10])
                except ValueError:
                    c = -1.0
                confs.append(c / 100.0 if c >= 0 else 0.0)
        record("CPU", "tesseract", t0, time.time())
        return " ".join(words), confs


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def vote(readings: dict[str, tuple[str, float]], weights: dict[str, float]):
    """readings: engine -> (value, confidence). Returns the resolution record.

    Weight of a candidate is the sum of (engine prior x confidence) over the
    engines that produced it, so two mediocre agreeing engines can outvote one
    confident outlier.
    """
    score: dict[str, float] = defaultdict(float)
    for engine, (value, conf) in readings.items():
        if value is None:
            continue
        base = engine.split(":")[0]
        score[value] += weights.get(base, 1.0) * conf
    if not score:
        return None
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    best, best_w = ranked[0]
    runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
    values = {v for v, _ in readings.values() if v is not None}
    return {
        "value": best,
        "weight": round(best_w, 4),
        "margin": round(best_w - runner_w, 4),
        "unanimous": len(values) == 1,
        "n_engines": len(values),
        "readings": {e: {"value": v, "confidence": round(c, 4)} for e, (v, c) in readings.items()},
    }


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #
def md_to_html(blocks_md, resolutions, title):
    """Render the page as standalone HTML.

    The VL model already emits HTML tables for table blocks, so those are passed
    through; the rest is converted with a small block-level markdown pass. Every
    number the vote touched is wrapped in a <span> carrying its readings, so a
    reviewer can hover a figure and see where it came from -- the audit trail the
    markdown output could only put in a separate file.
    """
    from html import escape

    # The VL model emits table blocks as OTSL (<fcel>/<nl>), not HTML. paddlex
    # ships the converter and it handles row/column spans, so reuse it rather
    # than re-deriving the grid here.
    try:
        from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html
    except Exception:
        convert_otsl_to_html = None

    by_value = defaultdict(list)
    for r in resolutions:
        by_value[r["value"]].append(r)

    def annotate(text):
        """Wrap resolved numbers so their provenance survives into the page."""
        def repl(m):
            val = normalise_number(m.group())
            hits = by_value.get(val)
            if not hits:
                return escape(m.group())
            r = hits[0]
            reads = "; ".join(f"{e}={v['value']}@{v['confidence']:.2f}"
                              for e, v in r["readings"].items())
            cls = "num"
            if r.get("suspect_lost_separator"):
                cls += " suspect"
            elif not r["unanimous"]:
                cls += " disputed"
            return (f'<span class="{cls}" title="{escape(reads)}">'
                    f'{escape(m.group())}</span>')
        return NUMBER_RE.sub(repl, text)

    parts = []
    for md in blocks_md:
        md = md.strip()
        if not md:
            continue
        if "<fcel>" in md or "<ecel>" in md:
            table = convert_otsl_to_html(md) if convert_otsl_to_html else None
            if table:
                # First row is the header in every table these documents produce.
                table = re.sub(r"<tr>(.*?)</tr>",
                               lambda m: "<thead><tr>"
                                         + m.group(1).replace("<td>", "<th>").replace("</td>", "</th>")
                                         + "</tr></thead><tbody>",
                               table, count=1)
                table = table.replace("</table>", "</tbody></table>")
                parts.append(annotate(table))
            else:
                parts.append(f"<pre>{escape(md)}</pre>")
            continue
        if md.lstrip().startswith("<"):     # already HTML
            parts.append(annotate(md))
            continue
        for para in md.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            heading = re.match(r"^(#{1,6})\s+(.*)$", para)
            if heading:
                lvl = len(heading.group(1))
                parts.append(f"<h{lvl}>{annotate(escape(heading.group(2)))}</h{lvl}>")
            elif all(l.lstrip().startswith(("- ", "* ")) for l in para.splitlines()):
                items = "".join(f"<li>{annotate(escape(l.lstrip()[2:]))}</li>"
                                for l in para.splitlines())
                parts.append(f"<ul>{items}</ul>")
            else:
                body = "<br>".join(annotate(escape(l)) for l in para.splitlines())
                parts.append(f"<p>{body}</p>")

    counts = {
        "total": len(resolutions),
        "disputed": sum(1 for r in resolutions if not r["unanimous"]),
        "suspect": sum(1 for r in resolutions if r.get("suspect_lost_separator")),
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ font: 16px/1.6 -apple-system, system-ui, sans-serif; max-width: 60rem;
         margin: 2rem auto; padding: 0 1rem; color: #111; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #bbb; padding: .35rem .6rem; text-align: right; }}
  th {{ background: #f2f2f2; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .num {{ border-bottom: 1px dotted #999; cursor: help; }}
  .num.disputed {{ background: #fff3cd; border-bottom-color: #b8860b; }}
  .num.suspect {{ background: #f8d7da; border-bottom-color: #c00; }}
  .legend {{ font-size: .85rem; color: #555; border-top: 1px solid #ddd;
             margin-top: 2rem; padding-top: .75rem; }}
  .legend span {{ padding: 0 .2rem; }}
</style></head><body>
{chr(10).join(parts)}
<div class="legend">
  {counts['total']} numbers resolved by three-engine vote &middot;
  <span class="num disputed">{counts['disputed']} disputed</span> &middot;
  <span class="num suspect">{counts['suspect']} possible lost separator</span>.
  Hover any underlined number for the individual engine readings.
</div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", nargs="?", default="table_sample.png")
    ap.add_argument("--lang", default="eng", help="Tesseract language(s)")
    ap.add_argument("--outdir", default="output/hybrid")
    ap.add_argument("--w-vl", type=float, default=1.0, help="engine prior: VL model")
    ap.add_argument("--w-ppocr", type=float, default=1.0, help="engine prior: PP-OCRv6")
    ap.add_argument("--w-tesseract", type=float, default=1.0, help="engine prior: Tesseract")
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--pad", type=int, default=6,
                    help="pixels of padding added before re-reading a numeric line")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML render")
    ap.add_argument("--vl-max-pixels", type=int, default=1024 * 28 * 28,
                    help="cap on pixels sent to the VL model; OCR engines still "
                         "see the full-resolution crop")
    args = ap.parse_args()

    global VL_MAX_PIXELS
    VL_MAX_PIXELS = args.vl_max_pixels
    weights = {"vl": args.w_vl, "ppocr": args.w_ppocr, "tesseract": args.w_tesseract}
    os.makedirs(args.outdir, exist_ok=True)

    page = cv2.imread(args.image)
    if page is None:
        sys.exit(f"cannot read image: {args.image}")
    t_start = time.time()

    # ---- layout: once, up front; both lanes consume it ----------------------
    from paddleocr import LayoutDetection

    t0 = time.time()
    layout = LayoutDetection(model_name="PP-DocLayoutV2")
    lay = next(iter(layout.predict(args.image)))
    record("CPU", "layout", t0, time.time())
    blocks = []
    for b in lay["boxes"]:
        x1, y1, x2, y2 = (int(v) for v in b["coordinate"])
        blocks.append({"label": b["label"], "score": b["score"], "box": (x1, y1, x2, y2)})
    blocks.sort(key=lambda b: (b["box"][1], b["box"][0]))
    print(f"layout: {len(blocks)} blocks "
          f"({sum(1 for b in blocks if b['label'] == 'table')} table)", flush=True)
    del layout  # free the layout predictor before the CPU lane loads its own

    def crop(box):
        x1, y1, x2, y2 = box
        return page[max(0, y1):y2, max(0, x1):x2]

    # ---- GPU lane: transcribe every non-image block -------------------------
    gpu_pool = ThreadPoolExecutor(max_workers=args.gpu_workers, thread_name_prefix="gpu")
    block_futures = {}
    for i, b in enumerate(blocks):
        if b["label"] in IMAGE_LABELS:
            continue
        block_futures[i] = gpu_pool.submit(vl_read, crop(b["box"]), b["label"])

    # ---- CPU lane: find and re-read numeric lines ---------------------------
    # Runs in its own thread so it overlaps the GPU lane above. Paddle only ever
    # touches this thread.
    numeric_lines: list[dict] = []

    def cpu_lane():
        eng = CpuEngines()
        for i, b in enumerate(blocks):
            if b["label"] in IMAGE_LABELS:
                continue
            block_img = crop(b["box"])
            if block_img.size == 0:
                continue
            try:
                polys = eng.lines(block_img)
            except Exception as exc:  # a bad crop shouldn't sink the page
                print(f"  det failed on block {i} ({b['label']}): {exc}", flush=True)
                continue
            boxes = []
            for poly in polys:
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                boxes.append((max(0, min(xs)), max(0, min(ys)), max(xs), max(ys)))
            for lx1, ly1, lx2, ly2 in dedupe_boxes(boxes):
                line_img = block_img[ly1:ly2, lx1:lx2]
                if line_img.size == 0 or line_img.shape[0] < 4 or line_img.shape[1] < 4:
                    continue
                p_text, p_score = eng.rec_line(line_img)
                if not numeric_tokens(p_text):
                    continue  # prose: the VL model handles it, no vote needed
                # Pad before re-reading: detection boxes clip descenders, and a
                # comma hangs below the baseline.
                pad = args.pad
                py1, py2 = max(0, ly1 - pad), min(block_img.shape[0], ly2 + pad)
                px1, px2 = max(0, lx1 - pad), min(block_img.shape[1], lx2 + pad)
                padded = block_img[py1:py2, px1:px2]
                variant_reads = []
                for vname, vimg in preprocess_variants(padded):
                    vp_text, vp_score = eng.rec_line(vimg)
                    variant_reads.append((f"ppocr:{vname}", vp_text, vp_score))
                    vt_text, vt_confs = eng.tess_line(vimg, args.lang)
                    vt_conf = (sum(vt_confs) / len(vt_confs)) if vt_confs else 0.0
                    variant_reads.append((f"tesseract:{vname}", vt_text, vt_conf))
                t_text, t_confs = eng.tess_line(line_img, args.lang)
                # Hand the crop to the GPU immediately: the VL re-read then runs
                # while this thread is still cropping and reading later lines,
                # instead of waiting for the whole CPU lane to finish.
                fut = gpu_pool.submit(vl_read, line_img, "text", 256)
                numeric_lines.append({
                    "variants": variant_reads,
                    "vl_future": fut,
                    "block": i,
                    "label": b["label"],
                    "line_box_in_page": (b["box"][0] + lx1, b["box"][1] + ly1,
                                         b["box"][0] + lx2, b["box"][1] + ly2),
                    "ppocr": {"text": p_text, "score": p_score},
                    "tesseract": {"text": t_text, "confs": t_confs},
                    "img": line_img,
                })

    cpu_thread = threading.Thread(target=cpu_lane, name="cpu-lane")
    t_lanes = time.time()
    cpu_thread.start()
    cpu_thread.join()   # GPU keeps draining its queue throughout
    block_texts = {}
    for i, fut in block_futures.items():
        try:
            block_texts[i] = fut.result(timeout=600)
        except Exception as exc:
            print(f"  VL failed on block {i}: {exc}", flush=True)
            block_texts[i] = ("", 0.0, [])
    lanes_wall = time.time() - t_lanes
    print(f"lanes done in {lanes_wall:.1f}s; {len(numeric_lines)} numeric lines", flush=True)

    # ---- fusion: collect the VL line re-reads the CPU lane already queued ---
    resolutions = []
    for nl in numeric_lines:
        v_text, v_conf, v_toks = nl["vl_future"].result(timeout=600)
        nl["vl_text"] = v_text
        suspect = any(suspect_lost_separator(t) for t in
                      (v_text, nl["ppocr"]["text"], nl["tesseract"]["text"]))
        t_mean = (sum(nl["tesseract"]["confs"]) / len(nl["tesseract"]["confs"])
                  if nl["tesseract"]["confs"] else 0.0)
        vl_conf_by_value = vl_token_confidences(v_toks, numeric_tokens(v_text))

        # Every independent look at this line becomes a voter: the two base
        # engines, the VL model, and each pre-processing variant.
        sources = [("vl", v_text, v_conf),
                   ("ppocr", nl["ppocr"]["text"], nl["ppocr"]["score"]),
                   ("tesseract", nl["tesseract"]["text"], t_mean)]
        sources += list(nl.get("variants", []))
        readings = [(name, numeric_tokens(txt), conf)
                    for name, txt, conf in sources if numeric_tokens(txt)]

        n, kept = reconcile(readings)
        aligned = len({len(r[1]) for r in readings}) == 1
        recovered = n < max((len(r[1]) for r in readings), default=0)

        for k in range(n):
            per_engine = {}
            for name, toks, conf in kept:
                if k >= len(toks):
                    continue
                c = vl_conf_by_value.get(toks[k], conf) if name == "vl" else conf
                per_engine[name] = (toks[k], c)
            r = vote(per_engine, weights)
            if r is None:
                continue
            r.update({
                "suspect_lost_separator": suspect,
                "separator_recovered": recovered,
                "raw_text": {
                    "vl": v_text,
                    "ppocr": nl["ppocr"]["text"],
                    "tesseract": nl["tesseract"]["text"],
                },
                "index_in_line": k,
                "block": nl["block"],
                "block_label": nl["label"],
                "box": nl["line_box_in_page"],
                "token_counts_agree": aligned,
            })
            resolutions.append(r)

    gpu_pool.shutdown(wait=True)

    # ---- report ------------------------------------------------------------
    page_md = []
    for i, b in enumerate(blocks):
        if i in block_texts:
            txt = block_texts[i][0].strip()
            if txt:
                page_md.append(txt)
    md_path = os.path.join(args.outdir, "page.md")
    with open(md_path, "w") as fh:
        fh.write("\n\n".join(page_md) + "\n")

    html_path = None
    if not args.no_html:
        html_path = os.path.join(args.outdir, "page.html")
        with open(html_path, "w") as fh:
            fh.write(md_to_html(page_md, resolutions, os.path.basename(args.image)))

    disputed = [r for r in resolutions if not r["unanimous"]]
    audit = {
        "image": args.image,
        "engine_priors": weights,
        "blocks": len(blocks),
        "numeric_lines": len(numeric_lines),
        "numbers_resolved": len(resolutions),
        "unanimous": sum(1 for r in resolutions if r["unanimous"]),
        "disputed": len(disputed),
        "token_count_mismatch_lines": sum(1 for r in resolutions if not r["token_counts_agree"]),
        "suspect_lost_separator": sum(1 for r in resolutions if r.get("suspect_lost_separator")),
        "wall_seconds": round(time.time() - t_start, 2),
        "separator_recovered": sum(1 for r in resolutions if r.get("separator_recovered")),
        "vl_usage": {
            "calls": _usage["calls"],
            "prompt_tokens": _usage["prompt_tokens"],
            "completion_tokens": _usage["completion_tokens"],
            "total_tokens": _usage["prompt_tokens"] + _usage["completion_tokens"],
        },
        "resolutions": resolutions,
    }
    audit_path = os.path.join(args.outdir, "numbers.json")
    with open(audit_path, "w") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    # lane overlap: how much GPU time landed inside the CPU lane's window
    cpu_span = [(a, b) for lane, _, a, b in _timeline if lane == "CPU"]
    gpu_span = [(a, b) for lane, _, a, b in _timeline if lane == "GPU"]
    def busy(spans):
        merged, total = [], 0.0
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        for a, b in merged:
            total += b - a
        return total, merged
    cpu_busy, cpu_m = busy(cpu_span)
    gpu_busy, gpu_m = busy(gpu_span)
    overlap = 0.0
    for a, b in cpu_m:
        for c, d in gpu_m:
            overlap += max(0.0, min(b, d) - max(a, c))

    u = audit["vl_usage"]
    wall = audit["wall_seconds"]
    print(f"\nVL usage: {u['calls']} calls, {u['prompt_tokens']} prompt + "
          f"{u['completion_tokens']} completion = {u['total_tokens']} tokens "
          f"({u['total_tokens'] / wall:.0f} tok/s over the page)")
    print(f"\nwrote {md_path}")
    if html_path:
        print(f"wrote {html_path}")
    print(f"wrote {audit_path}")
    print(f"\nnumbers: {len(resolutions)} resolved, {audit['unanimous']} unanimous, "
          f"{len(disputed)} disputed")
    print(f"wall {audit['wall_seconds']}s | CPU busy {cpu_busy:.1f}s | "
          f"GPU busy {gpu_busy:.1f}s | overlapped {overlap:.1f}s")
    suspects = [r for r in resolutions if r.get("suspect_lost_separator")]
    if suspects:
        print(f"\n{len(suspects)} number(s) on lines where a decimal separator may have been "
              f"lost -- unanimity here is NOT confirmation:")
        seen = set()
        for r in suspects:
            key = r["raw_text"]["ppocr"]
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{r['block_label']}] ppocr read: {key!r}")

    rec_n = audit["separator_recovered"]
    if rec_n:
        print(f"{rec_n} number(s) had a separator recovered by a pre-processing variant")

    if disputed:
        print("\ndisputed numbers (engine -> reading @ confidence):")
        for r in disputed[:15]:
            rd = ", ".join(f"{e}={v['value']}@{v['confidence']:.2f}"
                           for e, v in r["readings"].items())
            print(f"  [{r['block_label']}] -> {r['value']} "
                  f"(margin {r['margin']:.2f}) | {rd}")


if __name__ == "__main__":
    main()
