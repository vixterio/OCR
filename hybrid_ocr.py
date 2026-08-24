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
def vl_read(image_bgr, label: str = "text", max_tokens: int = 4096, timeout: int = 300):
    """Transcribe one crop with the VL model. Returns (text, mean_conf, tokens).

    tokens is [(token_text, confidence)], derived from per-token logprobs, which
    is what lets a single digit inside a long string carry its own confidence.
    """
    import urllib.request

    ok, buf = cv2.imencode(".png", image_bgr)
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
        score[value] += weights.get(engine, 1.0) * conf
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
    args = ap.parse_args()

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
                t_text, t_confs = eng.tess_line(line_img, args.lang)
                # Hand the crop to the GPU immediately: the VL re-read then runs
                # while this thread is still cropping and reading later lines,
                # instead of waiting for the whole CPU lane to finish.
                fut = gpu_pool.submit(vl_read, line_img, "text", 256)
                numeric_lines.append({
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
        p_tok = numeric_tokens(nl["ppocr"]["text"])
        t_tok = numeric_tokens(nl["tesseract"]["text"])
        v_tok = numeric_tokens(v_text)
        vl_conf_by_value = vl_token_confidences(v_toks, v_tok)
        t_mean = (sum(nl["tesseract"]["confs"]) / len(nl["tesseract"]["confs"])
                  if nl["tesseract"]["confs"] else 0.0)

        # Vote position by position. Engines that disagree on how many numbers
        # the line holds are recorded as such rather than force-aligned.
        n = max(len(p_tok), len(t_tok), len(v_tok))
        aligned = len({len(p_tok), len(t_tok), len(v_tok)}) == 1
        for k in range(n):
            readings = {}
            if k < len(v_tok):
                readings["vl"] = (v_tok[k], vl_conf_by_value.get(v_tok[k], v_conf))
            if k < len(p_tok):
                readings["ppocr"] = (p_tok[k], nl["ppocr"]["score"])
            if k < len(t_tok):
                readings["tesseract"] = (t_tok[k], t_mean)
            r = vote(readings, weights)
            if r is None:
                continue
            r.update({
                "suspect_lost_separator": suspect,
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

    print(f"\nwrote {md_path}")
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

    if disputed:
        print("\ndisputed numbers (engine -> reading @ confidence):")
        for r in disputed[:15]:
            rd = ", ".join(f"{e}={v['value']}@{v['confidence']:.2f}"
                           for e, v in r["readings"].items())
            print(f"  [{r['block_label']}] -> {r['value']} "
                  f"(margin {r['margin']:.2f}) | {rd}")


if __name__ == "__main__":
    main()
