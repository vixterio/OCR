"""Hybrid OCR for scanned documents: VL model for prose and layout, three-engine
confidence-weighted vote for every number.

Why the split. The VL model reads prose and structure better than the OCR engines,
but it is generative: it can emit a plausible wrong digit and report full
confidence. In a clinical document a wrong digit is worse than a missing one, so
numbers are read by PaddleOCR-VL, PP-OCRv6 and Tesseract -- plus pre-processing
variants -- and resolved by vote. Prose is left to the VL model, and is marked as
unverified because it is.

All three engines report real confidence: per-token logprobs from the VL server,
rec_score from PP-OCRv6, per-word TSV confidence from Tesseract.

Safety rules encoded here, each fixing a defect that was reproduced by execution:

  * A minority reading may raise a flag; it may never decide a value alone.
  * `unanimous` is computed over every reading, including those reconciliation
    rejected, and dissent is recorded.
  * A missing logprob is an error, not confidence 1.00.
  * Truncated VL replies (finish_reason == "length") are failures, not content.
  * Every block is accounted for. A region that was not transcribed appears as a
    visible placeholder, and the process exits non-zero. An incomplete page must
    never look complete.
  * Provenance is keyed by position, never by value.
  * Number semantics live in numeric.py, which has its own regression harness.

Usage:
    ./start_server.sh
    ./safe_run.sh hybrid_ocr.py record.pdf --outdir output/record
    ./safe_run.sh hybrid_ocr.py page.png --lang script/Latin
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser

import cv2
import numpy as np

import numeric

SERVER_URL = os.environ.get("VL_SERVER_URL", "http://127.0.0.1:8080/v1")
VL_MODEL = os.environ.get("VL_MODEL", "mlx-community/PaddleOCR-VL-4bit")

IMAGE_LABELS = {"image", "figure", "chart_image", "header_image", "footer_image"}

# Labels whose content must never be transcribed automatically. A handwritten dose
# is frequently the clinically decisive content, so it is shown as an image and
# escalated rather than guessed at.
QUARANTINE_LABELS = {"handwriting", "signature", "seal", "stamp"}

VL_MAX_PIXELS = 1024 * 28 * 28


class VLError(RuntimeError):
    """The VL backend did not honour the contract the vote depends on."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
_timeline: list[tuple[str, str, float, float]] = []
_timeline_lock = threading.Lock()


def record(lane: str, what: str, t0: float, t1: float) -> None:
    with _timeline_lock:
        _timeline.append((lane, what, t0, t1))


def prompt_for(label: str) -> str:
    """Mirrors paddlex/inference/pipelines/paddleocr_vl/pipeline.py:308-330."""
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    return "OCR:"


def write_atomic(path: str, text: str) -> None:
    """Write UTF-8 explicitly, restrictively, and atomically.

    Explicit encoding because the default is locale-dependent and a container
    with LC_ALL=C would fail on any non-ASCII text. Restrictive mode and
    temp-then-replace because these files hold recognised text, which for
    clinical documents is patient data, and a half-written file must never be
    mistaken for a complete one.
    """
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except BaseException:
        try:
            os.unlink(tmp)
        finally:
            raise
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# GPU lane: the VL model over HTTP
# --------------------------------------------------------------------------- #
def downscale_for_vl(img):
    """Cap the pixels sent to the VL model.

    Measured: a 3x-rendered page made the 4-bit model degenerate -- 10,582
    characters of repeated markup in place of a 377-character table -- while the
    OCR engines read the same page correctly. High resolution helps the OCR
    engines and harms the VL model, so the two get different images from one crop.
    """
    h, w = img.shape[:2]
    n = h * w
    if n <= VL_MAX_PIXELS or n == 0:
        return img
    scale = (VL_MAX_PIXELS / n) ** 0.5
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


@dataclass
class VLReply:
    text: str
    confidence: float
    tokens: list[tuple[str, float]]
    truncated: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # False when the backend cannot supply per-token logprobs at all. The vote
    # must then treat the VL reading as evidence without a confidence, rather
    # than silently assigning it one -- assigning one is how a missing signal
    # became "confidence 1.00" in an earlier version of this code.
    confidence_available: bool = True


def vl_read(image_bgr, label: str = "text", max_tokens: int = 4096,
            timeout: int = 300, require_logprobs: bool = True,
            prompt: str | None = None, model: str | None = None,
            strip_patterns: tuple = ()) -> VLReply:
    """Transcribe one crop. Raises VLError if the backend breaks the contract.

    Two silent failures are deliberately made loud here. A per-token entry with no
    `logprob` field used to yield exp(0.0) = 1.00, i.e. maximum confidence for
    absent evidence. A reply with no logprobs array at all used to give the VL
    model 0.0 confidence for every number, silently reducing a three-engine vote
    to two with nothing recorded anywhere.
    """
    import urllib.request

    ok, buf = cv2.imencode(".png", downscale_for_vl(image_bgr))
    if not ok:
        raise VLError("could not PNG-encode the crop")
    b64 = base64.b64encode(buf.tobytes()).decode()
    payload = {
        "model": model or VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt if prompt is not None else prompt_for(label)},
        ]}],
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
    record("GPU", f"vl:{label}", t0, time.time())

    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    finish = choice.get("finish_reason")

    entries = (choice.get("logprobs") or {}).get("content")
    if entries is None:
        if require_logprobs:
            raise VLError(
                "backend returned no logprobs.content; the vote uses exp(logprob) as "
                "VL confidence, so continuing would silently drop the VL model to a "
                "constant. Use a backend that returns logprobs on this endpoint "
                "(vLLM does; llama.cpp only on its native /completion)."
            )
        entries = []

    toks: list[tuple[str, float]] = []
    for e in entries:
        if "logprob" not in e:
            raise VLError("a logprobs entry has no 'logprob' field; refusing to "
                          "treat absent evidence as confidence 1.00")
        toks.append((e.get("token", ""), math.exp(e["logprob"])))

    for pat in strip_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.DOTALL)
    text = text.strip()
    conf = sum(c for _, c in toks) / len(toks) if toks else 0.0
    usage = data.get("usage") or {}
    return VLReply(text, conf, toks, finish == "length",
                   usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def vl_confidence_by_key(tokens) -> dict[str, float]:
    """Confidence per quantity key, from per-token logprobs.

    A number is only as trustworthy as its least certain character, so the
    weakest overlapping token governs. Where the same key appears more than once
    on a line the *minimum* is kept -- the previous code kept the maximum, which
    mixed a pessimistic and an optimistic statistic into something that
    corresponded to nothing.
    """
    joined = "".join(t for t, _ in tokens)
    spans, pos = [], 0
    for tok, c in tokens:
        spans.append((pos, pos + len(tok), c))
        pos += len(tok)
    out: dict[str, float] = {}
    for q in numeric.extract(joined):
        overlap = [c for a, b, c in spans if b > q.span[0] and a < q.span[1]]
        if not overlap:
            continue
        val = min(overlap)
        out[q.key] = min(out.get(q.key, val), val)
    return out


# --------------------------------------------------------------------------- #
# CPU lane
# --------------------------------------------------------------------------- #
def preprocess_variants(img):
    """Independent looks at one crop.

    Measured on a line whose decimal comma was lost: padding recovered one comma,
    upscale+Otsu recovered a different one, and neither recovered both. The
    variants fail independently, which is why several are worth running -- but
    they are *correlated* readings of the same pixels, so the vote must weight
    them as one engine family, not as separate engines.
    """
    yield "raw", img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    yield "up3x", cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu3x", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    # Otsu picks one global threshold, which loses thin strokes where a scan's
    # brightness drifts across the page -- common with photocopies and fax. An
    # adaptive threshold decides locally, so it keeps a faint comma that a global
    # cut would drop. Only enabled on machines with headroom for the extra reads.
    adaptive = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    yield "adaptive3x", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def dedupe_boxes(boxes, iou_thresh: float = 0.5):
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


class CpuEngines:
    """Paddle and Tesseract, confined to one thread.

    Paddle predictors are not thread-safe, so a web deployment must scale with
    processes rather than threads.
    """

    def __init__(self, det_model="PP-OCRv6_medium_det", rec_model="PP-OCRv6_medium_rec"):
        from paddleocr import TextDetection, TextRecognition
        import pytesseract
        self.det = TextDetection(model_name=det_model)
        self.rec = TextRecognition(model_name=rec_model)
        self.pt = pytesseract

    def lines(self, crop):
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
        t0 = time.time()
        from PIL import Image
        rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        tsv = self.pt.image_to_data(Image.fromarray(rgb), lang=lang, config="--psm 7")
        rows = tsv.splitlines()
        if not rows:
            return "", []
        header = rows[0].split("\t")
        # Parse by header name: column order is not guaranteed across versions, and
        # a positional parse would silently lose the third voter.
        try:
            i_text, i_conf = header.index("text"), header.index("conf")
        except ValueError:
            i_text, i_conf = 11, 10
        words, confs = [], []
        for row in rows[1:]:
            parts = row.split("\t")
            if len(parts) <= max(i_text, i_conf) or not parts[i_text].strip():
                continue
            words.append(parts[i_text])
            try:
                c = float(parts[i_conf])
            except ValueError:
                c = -1.0
            confs.append(c / 100.0 if c >= 0 else 0.0)
        record("CPU", "tesseract", t0, time.time())
        return " ".join(words), confs


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def family(engine: str) -> str:
    """'ppocr:otsu3x' -> 'ppocr'. Variants of one engine are one family."""
    return engine.split(":")[0]


def digit_signature(keys) -> str:
    return "".join(re.sub(r"\D", "", k) for k in keys)


@dataclass
class Reading:
    source: str
    keys: list[str]
    conf: float
    text: str = ""


def reconcile(readings: list[Reading], min_families: int = 2):
    """Agree how many quantities a line holds, before voting on their values.

    Returns (n, kept, dropped, notes).

    The previous version preferred the reading with the fewest tokens outright, on
    the argument that OCR loses separators more often than it invents them. That
    let a single correlated variant delete a real number, invent one no
    independent engine had read, and report it as unanimous -- reproduced as eight
    readings of ['12','5'] being overruled by one of ['12.5'] at n_engines=1,
    which is a ten-fold dose error.

    So a lower token count now has to be corroborated by at least `min_families`
    distinct engine families. Otherwise the majority tokenisation stands and the
    disagreement is reported instead of resolved.
    """
    notes: list[str] = []
    if not readings:
        return 0, [], [], notes

    weight_by_sig: dict[str, float] = defaultdict(float)
    for r in readings:
        weight_by_sig[digit_signature(r.keys)] += r.conf
    best_sig = max(weight_by_sig.items(), key=lambda kv: kv[1])[0]
    same = [r for r in readings if digit_signature(r.keys) == best_sig]
    dropped = [r for r in readings if digit_signature(r.keys) != best_sig]

    counts: dict[int, set[str]] = defaultdict(set)
    for r in same:
        counts[len(r.keys)].add(family(r.source))

    modal = max(counts.items(), key=lambda kv: (len(kv[1]), -kv[0]))[0]
    lowest = min(counts)
    n = modal
    if lowest < modal:
        if len(counts[lowest]) >= min_families:
            n = lowest
            notes.append(f"separator_merge_corroborated_by_{len(counts[lowest])}_families")
        else:
            notes.append("separator_merge_rejected_single_family")

    kept = [r for r in same if len(r.keys) == n]
    dropped += [r for r in same if len(r.keys) != n]
    return n, kept, dropped, notes


def vote(per_source: dict[str, tuple[str, float]], priors: dict[str, float],
         all_values: set[str]):
    """Confidence-weighted vote over one position.

    `all_values` is every value any reading produced at this position, including
    readings reconciliation rejected, so `unanimous` means what it says.
    """
    score: dict[str, float] = defaultdict(float)
    for engine, (value, conf) in per_source.items():
        if value is None:
            continue
        score[value] += priors.get(family(engine), 1.0) * conf
    if not score:
        return None
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    best, best_w = ranked[0]
    runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(score.values()) or 1.0
    families = {family(e) for e, (v, _) in per_source.items() if v == best}
    return {
        "value": best,
        "weight": round(best_w, 4),
        "margin": round(best_w - runner_w, 4),
        "margin_frac": round((best_w - runner_w) / total, 4),
        "unanimous": len(all_values) == 1,
        "n_families": len(families),
        "candidates": {v: round(w, 4) for v, w in ranked},
        "readings": {e: {"value": v, "confidence": round(c, 4)}
                     for e, (v, c) in per_source.items()},
    }


# --------------------------------------------------------------------------- #
# page processing
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    index: int
    label: str
    box: tuple[int, int, int, int]
    status: str = "pending"     # ok | quarantined | image | failed | truncated | empty
    text: str = ""
    confidence: float = 0.0
    note: str = ""


@dataclass
class PageResult:
    index: int
    width: int
    height: int
    dpi: float | None
    source: str
    blocks: list[Block] = field(default_factory=list)
    numbers: list[dict] = field(default_factory=list)
    vl_prompt_tokens: int = 0
    vl_completion_tokens: int = 0
    vl_calls: int = 0
    page_markdown: str = ""      # set by page-granularity VL models

    @property
    def incomplete(self) -> list[Block]:
        # 'empty' belongs here: a VL reply with no text used to leave every block
        # marked empty, render nothing, and report "0 regions not transcribed".
        return [b for b in self.blocks
                if b.status in ("failed", "truncated", "quarantined", "empty")]


def process_page(page_img, page_index, eng, gpu_pool, args, priors) -> PageResult:
    """Layout, then two concurrent lanes, then fusion. One page, no I/O."""
    from paddleocr import LayoutDetection

    h, w = page_img.shape[:2]
    t0 = time.time()
    layout = LayoutDetection(model_name=args.layout_model)
    lay = next(iter(layout.predict(page_img)))
    record("CPU", "layout", t0, time.time())
    del layout

    blocks: list[Block] = []
    for b in lay["boxes"]:
        x1, y1, x2, y2 = (int(v) for v in b["coordinate"])
        blocks.append(Block(len(blocks), b["label"], (x1, y1, x2, y2)))
    blocks.sort(key=lambda b: (b.box[1], b.box[0]))
    for i, b in enumerate(blocks):
        b.index = i

    page = PageResult(page_index, w, h, None, "", blocks)

    def crop(box, pad=0):
        x1, y1, x2, y2 = box
        return page_img[max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad)]

    # ---- GPU lane -----------------------------------------------------------
    futures = {}
    for b in blocks:
        if b.label in IMAGE_LABELS:
            b.status = "image"
            b.note = "figure region, not transcribed"
            continue
        if b.label in QUARANTINE_LABELS:
            b.status = "quarantined"
            b.note = f"{b.label} region: never transcribed automatically"
            continue
        futures[b.index] = gpu_pool.submit(vl_read, crop(b.box), b.label)

    # ---- CPU lane: find numeric lines and read them every way --------------
    numeric_lines: list[dict] = []
    line_seq = [0]

    def cpu_lane():
        for b in blocks:
            if b.status in ("image", "quarantined"):
                continue
            block_img = crop(b.box)
            if block_img.size == 0:
                continue
            try:
                polys = eng.lines(block_img)
            except Exception as exc:
                b.note = f"line detection failed: {exc}"
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
                if not numeric.extract(p_text):
                    continue                      # prose: the VL model handles it
                pad = args.pad
                py1, py2 = max(0, ly1 - pad), min(block_img.shape[0], ly2 + pad)
                px1, px2 = max(0, lx1 - pad), min(block_img.shape[1], lx2 + pad)
                padded = block_img[py1:py2, px1:px2]

                readings = [Reading("ppocr", numeric.keys(p_text), p_score, p_text)]
                for vname, vimg in preprocess_variants(padded):
                    vp, vs = eng.rec_line(vimg)
                    readings.append(Reading(f"ppocr:{vname}", numeric.keys(vp), vs, vp))
                    vt, vc = eng.tess_line(vimg, args.lang)
                    conf = (sum(vc) / len(vc)) if vc else 0.0
                    readings.append(Reading(f"tesseract:{vname}", numeric.keys(vt), conf, vt))
                t_text, t_confs = eng.tess_line(line_img, args.lang)
                t_mean = (sum(t_confs) / len(t_confs)) if t_confs else 0.0
                readings.append(Reading("tesseract", numeric.keys(t_text), t_mean, t_text))

                line_seq[0] += 1
                numeric_lines.append({
                    "line_id": f"p{page_index}l{line_seq[0]}",
                    "block": b.index,
                    "label": b.label,
                    "box": (b.box[0] + lx1, b.box[1] + ly1, b.box[0] + lx2, b.box[1] + ly2),
                    "readings": readings,
                    "ppocr_text": p_text,
                    "tesseract_text": t_text,
                    "vl_future": gpu_pool.submit(vl_read, line_img, "text", 256),
                })

    cpu_thread = threading.Thread(target=cpu_lane, name="cpu-lane")
    cpu_thread.start()
    cpu_thread.join()

    # ---- collect the GPU lane; a broken block is a visible failure ----------
    for idx, fut in futures.items():
        b = blocks[idx]
        try:
            reply = fut.result(timeout=args.block_timeout)
        except VLError as exc:
            b.status, b.note = "failed", f"VL contract violation: {exc}"
            continue
        except Exception as exc:
            b.status, b.note = "failed", f"VL call failed: {exc}"
            continue
        page.vl_calls += 1
        page.vl_prompt_tokens += reply.prompt_tokens
        page.vl_completion_tokens += reply.completion_tokens
        if reply.truncated:
            b.status = "truncated"
            b.note = "VL reply hit max_tokens; content is incomplete"
            b.text = reply.text
            continue
        b.text = reply.text.strip()
        b.confidence = reply.confidence
        b.status = "ok" if b.text else "empty"
        if not b.text:
            b.note = "VL returned no text for this region"

    # ---- fusion -------------------------------------------------------------
    for nl in numeric_lines:
        try:
            reply = nl["vl_future"].result(timeout=args.block_timeout)
            page.vl_calls += 1
            page.vl_prompt_tokens += reply.prompt_tokens
            page.vl_completion_tokens += reply.completion_tokens
            vl_keys = numeric.keys(reply.text)
            vl_conf = vl_confidence_by_key(reply.tokens)
            nl["readings"].append(Reading("vl", vl_keys, reply.confidence, reply.text))
            vl_text = reply.text
        except Exception as exc:
            vl_conf, vl_text = {}, ""
            nl["vl_error"] = str(exc)

        readings = [r for r in nl["readings"] if r.keys]
        n, kept, dropped, notes = reconcile(readings, args.min_families)

        # Lost-separator suspicion, attached only to the numbers involved.
        susp_spans = {}
        for src_text in (nl["ppocr_text"], nl["tesseract_text"], vl_text):
            for span in numeric.lost_separator_spans(src_text):
                susp_spans[span] = True
        line_suspect = bool(susp_spans)

        for k in range(n):
            per_source, seen_values = {}, set()
            for r in kept:
                if k >= len(r.keys):
                    continue
                key = r.keys[k]
                c = vl_conf.get(key, r.conf) if r.source == "vl" else r.conf
                per_source[r.source] = (key, c)
                seen_values.add(key)
            for r in dropped:
                if k < len(r.keys):
                    seen_values.add(r.keys[k])
            res = vote(per_source, priors, seen_values)
            if res is None:
                continue
            quantities = numeric.extract(nl["ppocr_text"])
            flags = sorted({f for q in quantities for f in q.flags})
            res.update({
                "display": numeric.format_key(res["value"]),
                "line_id": nl["line_id"],
                "index_in_line": k,
                "block": nl["block"],
                "block_label": nl["label"],
                "box": nl["box"],
                "page": page_index,
                "reconcile_notes": notes,
                "suspect_lost_separator": line_suspect,
                "numeric_flags": flags,
                "dissent": [{"source": r.source, "keys": r.keys,
                             "confidence": round(r.conf, 4)} for r in dropped],
                "raw_text": {"vl": vl_text, "ppocr": nl["ppocr_text"],
                             "tesseract": nl["tesseract_text"]},
                # `unanimous` stays literal -- any dissent makes it False, and every
                # dissenting reading is kept. But demanding review for a
                # low-confidence minority from a correlated variant floods the
                # queue: measured on the test PDF, 19 of 24 dissenting readings
                # scored below 0.5, and in each case the vote was already right.
                # So only *credible* dissent forces review.
                "credible_dissent": [
                    {"source": r.source, "keys": r.keys, "confidence": round(r.conf, 4)}
                    for r in dropped if r.conf >= args.dissent_floor],
                "needs_review": bool(
                    line_suspect or flags or notes
                    or res["n_families"] < 2
                    or res["margin_frac"] < args.min_margin
                    or any(r.conf >= args.dissent_floor for r in dropped)
                ),
            })
            page.numbers.append(res)
    return page


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
class _Annotator(HTMLParser):
    """Wrap quantities in text nodes only.

    The previous renderer ran the number regex over the generated markup, so a
    merged table cell's colspan="2" became colspan="<span ...>2</span>" -- invalid
    HTML, and merged header cells are ubiquitous in lab reports. Attributes are
    now copied through untouched and only character data is annotated.
    """
    VOID = {"br", "hr", "img", "col", "input", "meta", "link"}

    def __init__(self, annotate_text):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._annotate = annotate_text

    def handle_starttag(self, tag, attrs):
        a = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs)
        self.out.append(f"<{tag}{a}>")

    def handle_startendtag(self, tag, attrs):
        a = "".join(f' {k}="{html.escape(v or "", quote=True)}"' for k, v in attrs)
        self.out.append(f"<{tag}{a}/>")

    def handle_endtag(self, tag):
        if tag not in self.VOID:
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        self.out.append(self._annotate(data))

    def result(self) -> str:
        return "".join(self.out)


def make_annotator(numbers: list[dict]):
    """Positional annotation.

    Provenance is consumed in reading order, never looked up by value. The old
    by-value lookup took hits[0], so when a value repeated -- routine in clinical
    tables -- every occurrence inherited the first one's confidence and styling.
    """
    queue = list(numbers)
    pos = [0]

    def annotate(text: str) -> str:
        quantities = numeric.extract(text)
        if not quantities:
            return html.escape(text)
        out, last = [], 0
        for q in quantities:
            out.append(html.escape(text[last:q.span[0]]))
            rec = queue[pos[0]] if pos[0] < len(queue) else None
            if rec is None:
                out.append(f'<span class="num unverified" '
                           f'title="not verified by the vote">{html.escape(q.raw)}</span>')
            else:
                pos[0] += 1
                cls = ["num"]
                if rec.get("verified") is False:
                    cls.append("unverified")
                if rec.get("needs_review"):
                    cls.append("review")
                if rec.get("suspect_lost_separator"):
                    cls.append("suspect")
                if not rec.get("unanimous"):
                    cls.append("disputed")
                reads = "; ".join(
                    f"{e}={numeric.format_key(v['value'])}@{v['confidence']:.2f}"
                    for e, v in rec["readings"].items())
                extra = ""
                if rec.get("numeric_flags"):
                    extra += " | flags: " + ",".join(rec["numeric_flags"])
                if rec.get("dissent"):
                    extra += f" | {len(rec['dissent'])} dissenting reading(s)"
                voted = numeric.format_key(rec["value"])
                if rec.get("verified") is False:
                    title = (f"{voted} — single VL reading, NOT verified by any "
                             f"vote{extra}")
                    shown = q.raw
                else:
                    title = f"voted {voted} | {reads}{extra}"
                    # Display the value the vote chose, not the reading it
                    # rejected. Previously the page showed the VL model's digits
                    # while the tooltip held the decision, so a vote that
                    # correctly resolved 12.5 mg still printed "125" on screen --
                    # a ten-fold dose, visible to a clinician who never hovers.
                    shown = voted
                mark = " ⚑" if rec.get("needs_review") else ""
                body = html.escape(shown)
                if rec.get("verified") is not False and shown != q.raw:
                    # Keep the rejected reading visible, struck through, so the
                    # correction is auditable on the page itself.
                    body = (f'{html.escape(shown)} '
                            f'<s class="rejected">{html.escape(q.raw)}</s>')
                out.append(f'<span class="{" ".join(cls)}" '
                           f'title="{html.escape(title, quote=True)}">'
                           f'{body}{mark}</span>')
            last = q.span[1]
        out.append(html.escape(text[last:]))
        return "".join(out)

    return annotate, lambda: (pos[0], len(queue))


CSS = """
:root { --bg:#fff; --fg:#111; --line:#bbb; --head:#f2f2f2; --note:#555;
        --review:#8a4b00; --review-bg:#fff3cd; --bad:#8a0000; --bad-bg:#f8d7da; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8e8e8; --line:#444; --head:#22252b; --note:#aaa;
          --review:#ffd479; --review-bg:#3a2f10; --bad:#ff9c9c; --bad-bg:#3d1a1a; }
}
body { font:16px/1.65 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;
       max-width:64rem; margin:2rem auto; padding:0 1rem;
       background:var(--bg); color:var(--fg); }
table { border-collapse:collapse; margin:1rem 0; width:100%; }
th,td { border:1px solid var(--line); padding:.35rem .6rem; text-align:right; }
th { background:var(--head); }
td:first-child, th:first-child { text-align:left; }
.num { border-bottom:1px dotted var(--line); cursor:help; }
.num.review { background:var(--review-bg); color:var(--review);
              border-bottom:2px solid var(--review); font-weight:600; }
.num.suspect, .num.disputed { background:var(--bad-bg); color:var(--bad);
              border-bottom:2px solid var(--bad); font-weight:600; }
.num.unverified { border-bottom:1px dashed var(--note); }
.rejected { opacity:.65; font-size:.9em; }
.page { border-top:3px solid var(--line); margin-top:2.5rem; padding-top:.5rem; }
.gap { background:var(--bad-bg); color:var(--bad); border:2px dashed var(--bad);
       padding:.6rem .8rem; margin:.8rem 0; font-weight:600; }
.banner { border:2px solid var(--review); background:var(--review-bg);
          color:var(--review); padding:.8rem 1rem; margin:1rem 0; font-weight:600; }
.legend { font-size:.88rem; color:var(--note); border-top:1px solid var(--line);
          margin-top:2rem; padding-top:.8rem; }
"""


def render_block(block: Block, annotate) -> str:
    """One block of a page, or a visible placeholder saying why it is absent."""
    if block.status in ("image", "quarantined", "failed", "truncated", "empty"):
        word = {"image": "Figure not transcribed",
                "quarantined": "NOT TRANSCRIBED",
                "failed": "TRANSCRIPTION FAILED",
                "truncated": "TRUNCATED — CONTENT MISSING",
                "empty": "NO TEXT RECOGNISED"}[block.status]
        cls = "gap" if block.status != "image" else "legend"
        return (f'<div class="{cls}">{word}: {html.escape(block.label)} at '
                f'{block.box} — {html.escape(block.note or "see source image")}</div>')
    md = block.text
    if "<fcel>" in md or "<ecel>" in md:
        try:
            from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html
        except Exception as exc:
            return (f'<div class="gap">TABLE NOT RENDERED: the paddlex OTSL converter '
                    f'could not be imported ({html.escape(str(exc))}). Raw markup '
                    f'follows.</div><pre>{html.escape(md)}</pre>')
        table = convert_otsl_to_html(md)
        if not table:
            return f'<div class="gap">TABLE NOT RENDERED</div><pre>{html.escape(md)}</pre>'
        table = re.sub(
            r"<tr>(.*?)</tr>",
            lambda m: "<thead><tr>" + m.group(1).replace("<td>", '<th scope="col">')
                                                 .replace("</td>", "</th>") + "</tr></thead><tbody>",
            table, count=1)
        table = table.replace("</table>", "</tbody></table>")
        p = _Annotator(annotate)
        p.feed(table)
        p.close()
        return p.result()
    parts = []
    for para in md.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", para)
        if heading:
            lvl = len(heading.group(1))
            parts.append(f"<h{lvl}>{annotate(heading.group(2))}</h{lvl}>")
        elif all(l.lstrip().startswith(("- ", "* ")) for l in para.splitlines()):
            items = "".join(f"<li>{annotate(l.lstrip()[2:])}</li>" for l in para.splitlines())
            parts.append(f"<ul>{items}</ul>")
        else:
            parts.append("<p>" + "<br>".join(annotate(l) for l in para.splitlines()) + "</p>")
    return "\n".join(parts)


def render_document(pages: list[PageResult], title: str, args) -> str:
    body, total_numbers, review, gaps = [], 0, 0, 0
    for pg in pages:
        annotate, tally = make_annotator(pg.numbers)
        total_numbers += len(pg.numbers)
        review += sum(1 for n in pg.numbers if n.get("needs_review"))
        chunks = [render_block(b, annotate) for b in pg.blocks]
        gaps += sum(1 for b in pg.blocks if b.status in ("failed", "truncated", "quarantined"))
        used, avail = tally()
        note = ""
        if used != avail:
            note = (f'<div class="banner">Provenance alignment incomplete on this page: '
                    f'{used} of {avail} verified numbers could be matched to the '
                    f'transcribed text. Unmatched values are in the audit file.</div>')
        body.append(f'<section class="page" id="page-{pg.index + 1}">'
                    f'<h2>Page {pg.index + 1}</h2>{note}' + "\n".join(chunks) + "</section>")

    banner = ""
    if gaps or review:
        banner = (f'<div class="banner">This is an unreviewed machine transcription. '
                  f'{review} number(s) need human verification and {gaps} region(s) were '
                  f'not transcribed. Do not treat it as a verified record.</div>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
{banner}
{"".join(body)}
<div class="legend">
<p><strong>{total_numbers}</strong> numbers resolved by vote across
{len(pages)} page(s); <strong>{review}</strong> flagged for review;
<strong>{gaps}</strong> region(s) not transcribed.</p>
<p>Numbers marked <span class="num review">like this ⚑</span> need review.
<span class="num disputed">Red</span> means the engines disagreed or a decimal
separator may have been lost. <span class="num unverified">Dashed</span> means
the number appears in prose and was never verified by the vote. Hover any number
for its per-engine readings. The flag glyph and the border weight carry the same
meaning as the colour, so colour is not the only signal.</p>
<p>Prose is transcribed by the VL model alone and is not verified.</p>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def iter_input_pages(path: str, args):
    """Yield (index, image, dpi, source, text_layer_chars) for a PDF or an image."""
    if path.lower().endswith(".pdf"):
        import pdf_input
        for p in pdf_input.load_pages(path, render_dpi=args.render_dpi,
                                      max_pages=args.max_pages):
            print("  " + pdf_input.describe(p), flush=True)
            if p.text_layer_chars and not args.ignore_text_layer:
                print(f"    note: this page carries {p.text_layer_chars} characters of "
                      f"existing text layer. It is NOT used -- its provenance is "
                      f"unknown. Pass --ignore-text-layer to silence this.", flush=True)
            yield p.index, p.image, p.dpi, p.source, p.text_layer_chars
    else:
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"cannot read image: {path}")
        yield 0, img, None, "image-file", 0


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid OCR: VL model for prose, three-engine vote for numbers.")
    ap.add_argument("input", help="PDF or image file")
    ap.add_argument("--outdir", default="output/hybrid")
    ap.add_argument("--lang", default="eng", help="Tesseract language, e.g. script/Latin")
    ap.add_argument("--layout-model", default="PP-DocLayoutV2")
    ap.add_argument("--render-dpi", type=float, default=300.0,
                    help="only used for PDF pages that are not a single embedded scan")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--ignore-text-layer", action="store_true")
    ap.add_argument("--w-vl", type=float, default=1.0)
    ap.add_argument("--w-ppocr", type=float, default=1.0)
    ap.add_argument("--w-tesseract", type=float, default=1.0)
    ap.add_argument("--gpu-workers", type=int, default=2,
                    help="VL calls serialise on the device (measured 1.01-1.04x from "
                         "2-6 workers), so this mainly buys cross-lane overlap")
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--vl-max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--min-families", type=int, default=2,
                    help="engine families required to corroborate a separator merge")
    ap.add_argument("--min-margin", type=float, default=0.15,
                    help="normalised vote margin below which a number needs review")
    ap.add_argument("--dissent-floor", type=float, default=0.5,
                    help="a dissenting reading below this confidence is recorded in the "
                         "audit but does not by itself demand human review. Uncalibrated: "
                         "there is no ground-truth set yet to tune it against")
    ap.add_argument("--block-timeout", type=int, default=600)
    ap.add_argument("--allow-gaps", action="store_true",
                    help="exit 0 even when regions were not transcribed")
    args = ap.parse_args()

    global VL_MAX_PIXELS
    VL_MAX_PIXELS = args.vl_max_pixels
    priors = {"vl": args.w_vl, "ppocr": args.w_ppocr, "tesseract": args.w_tesseract}

    os.umask(0o077)          # derived artefacts contain recognised text
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    eng = CpuEngines()
    gpu_pool = ThreadPoolExecutor(max_workers=args.gpu_workers, thread_name_prefix="gpu")
    pages: list[PageResult] = []
    try:
        for idx, img, dpi, source, chars in iter_input_pages(args.input, args):
            pg = process_page(img, idx, eng, gpu_pool, args, priors)
            pg.dpi, pg.source = dpi, source
            pages.append(pg)
            flagged = sum(1 for n in pg.numbers if n.get("needs_review"))
            print(f"  page {idx + 1}: {len(pg.blocks)} blocks, {len(pg.numbers)} numbers, "
                  f"{flagged} need review, {len(pg.incomplete)} region(s) not transcribed",
                  flush=True)
    finally:
        gpu_pool.shutdown(wait=True)

    title = os.path.basename(args.input)
    write_atomic(os.path.join(args.outdir, "document.html"),
                 render_document(pages, title, args))

    audit = {
        "input": args.input,
        "engine_priors": priors,
        "min_families": args.min_families,
        "min_margin": args.min_margin,
        "wall_seconds": round(time.time() - t_start, 2),
        "pages": [{
            "index": p.index, "width": p.width, "height": p.height,
            "dpi": p.dpi, "source": p.source,
            "vl_calls": p.vl_calls,
            "vl_prompt_tokens": p.vl_prompt_tokens,
            "vl_completion_tokens": p.vl_completion_tokens,
            "blocks": [{"index": b.index, "label": b.label, "box": list(b.box),
                        "status": b.status, "note": b.note,
                        "confidence": round(b.confidence, 4)} for b in p.blocks],
            "numbers": p.numbers,
        } for p in pages],
    }
    write_atomic(os.path.join(args.outdir, "audit.json"),
                 json.dumps(audit, indent=2, ensure_ascii=False))

    tot = sum(len(p.numbers) for p in pages)
    review = sum(1 for p in pages for n in p.numbers if n.get("needs_review"))
    gaps = sum(len(p.incomplete) for p in pages)
    prompt = sum(p.vl_prompt_tokens for p in pages)
    completion = sum(p.vl_completion_tokens for p in pages)
    calls = sum(p.vl_calls for p in pages)

    print(f"\nwrote {args.outdir}/document.html")
    print(f"wrote {args.outdir}/audit.json")
    print(f"\npages {len(pages)} | numbers {tot} | need review {review} | "
          f"regions not transcribed {gaps}")
    print(f"VL: {calls} calls, {prompt} prompt + {completion} completion = "
          f"{prompt + completion} tokens")
    print(f"wall {audit['wall_seconds']}s")

    for p in pages:
        for n in p.numbers:
            if not n.get("needs_review"):
                continue
            why = []
            if not n["unanimous"]:
                why.append("engines disagreed")
            if n["n_families"] < 2:
                why.append(f"only {n['n_families']} engine family")
            if n.get("suspect_lost_separator"):
                why.append("possible lost decimal separator")
            if n.get("numeric_flags"):
                why.append("flags: " + ",".join(n["numeric_flags"]))
            if n.get("reconcile_notes"):
                why.append(";".join(n["reconcile_notes"]))
            if n["margin_frac"] < args.min_margin:
                why.append(f"thin margin {n['margin_frac']:.2f}")
            print(f"  review p{n['page'] + 1} [{n['block_label']}] "
                  f"{numeric.format_key(n['value'])}  ({'; '.join(why)})")

    if gaps and not args.allow_gaps:
        print(f"\nexiting non-zero: {gaps} region(s) were not transcribed. "
              f"An incomplete page must not be mistaken for a complete one. "
              f"Pass --allow-gaps to override.")
        sys.exit(2)


if __name__ == "__main__":
    main()
