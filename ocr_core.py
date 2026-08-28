"""Shared OCR engine for all six modes.

The six modes differ in exactly two ways: which VL model transcribes, and whether
the OCR engines verify the numbers. Everything else -- layout, coverage
accounting, the CPU lane, the vote, the HTML -- is common, so it lives here once
rather than in six near-identical scripts.

One structural difference between VL families is handled explicitly:

  * `granularity == "block"` (PaddleOCR-VL): a layout pass hands the model one
    region at a time with a terse task prompt. N calls per page.
  * `granularity == "page"` (DeepSeek-OCR-2, Qwen3.5-VL): the whole page goes in
    one call and the model produces its own Markdown. 1 call per page.

Layout still runs in page mode, even though the VL model does not need it, because
it is what identifies handwriting and signature regions to quarantine and what
makes coverage accounting possible. A page model that silently skipped a
handwritten dose would be worse than a slower one that flags it.
"""
from __future__ import annotations

import html
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2

import numeric
from hybrid_ocr import (CSS, Block, CpuEngines, PageResult, Reading, VLError,
                        dedupe_boxes, make_annotator, preprocess_variants,
                        reconcile, record, vl_confidence_by_key, vl_read,
                        write_atomic, _Annotator, IMAGE_LABELS, QUARANTINE_LABELS,
                        downscale_for_vl, vote)
import hybrid_ocr


def block_prompt(spec, label: str) -> str:
    """Task prompt for one block, for block-granularity models."""
    bp = spec.block_prompts or {}
    if label == "table" and "table" in bp:
        return bp["table"]
    if label == "chart" and "chart" in bp:
        return bp["chart"]
    if "formula" in label and label != "formula_number" and "formula" in bp:
        return bp["formula"]
    return bp.get("default", spec.line_prompt or "OCR:")


def _slot(slots: list, text: str) -> str:
    """Reserve a placeholder for text, keyed by index.

    The previous scheme wrapped text in `{{ }}` and substituted with a non-greedy
    regex, so any transcribed text containing `}}` closed the slot early and
    everything up to the next `{{` was copied into the HTML unescaped. A scanned
    referral containing `}}<img src=x onerror=...>` would then execute in the
    reviewer's browser. Indices cannot collide with document content.
    """
    slots.append(text)
    return f"\x00SLOT{len(slots) - 1}\x00"


def md_tables_to_html(md: str) -> str:
    """Render Markdown pipe tables, headings, lists and paragraphs to HTML.

    DeepSeek-OCR-2 and Qwen3.5-VL emit Markdown, not the OTSL that PaddleOCR-VL
    produces, so page-granularity output needs its own renderer. Kept
    deliberately small: this is a transcription, not a general Markdown document.
    """
    lines = md.split("\n")
    out, i, slots = [], 0, []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # A pipe table: header row, separator row of dashes, then body rows.
        if (stripped.startswith("|") and i + 1 < len(lines)
                and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1])):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            head = cells(stripped)
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(cells(lines[i].strip()))
                i += 1
            th = "".join(f'<th scope="col">{_slot(slots, c)}</th>' for c in head)
            rows = []
            for r in body:
                # Pad or truncate to the header width, and say so, rather than
                # letting a short row shift every value under the wrong heading.
                cells_out = list(r[:len(head)]) + [""] * max(0, len(head) - len(r))
                flag = "" if len(r) == len(head) else (
                    ' data-cell-count-mismatch="true"')
                rows.append(f"<tr{flag}>" + "".join(
                    f"<td>{_slot(slots, c)}</td>" for c in cells_out) + "</tr>")
            rows = "".join(rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_slot(slots, m.group(2))}</h{lvl}>")
            i += 1
            continue
        if stripped.startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].strip().startswith(("- ", "* ")):
                items.append(f"<li>{_slot(slots, lines[i].strip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        para = []
        while (i < len(lines) and lines[i].strip()
               and not lines[i].strip().startswith(("#", "- ", "* "))
               and not (lines[i].strip().startswith("|") and para)):
            para.append(lines[i].strip())
            i += 1
        if not para:
            # A '|' row that was not accepted as a table (no separator row, or a
            # reply truncated mid-table) used to reach here and match nothing, so
            # `i` never advanced and this looped forever emitting empty
            # paragraphs until the watchdog killed the run. Emit it as text and
            # move on.
            para = [lines[i].strip()]
            i += 1
        out.append("<p>" + "<br>".join(_slot(slots, p) for p in para) + "</p>")
    return "\n".join(out), slots


_PLACEHOLDER = re.compile(r"\x00SLOT(\d+)\x00")


def fill_placeholders(markup: str, slots: list, annotate) -> str:
    """Substitute reserved slots with annotated, escaped text.

    Text reaches the page only through `annotate`, which escapes it, so no
    transcribed character can close a tag or an attribute.
    """
    return _PLACEHOLDER.sub(lambda m: annotate(slots[int(m.group(1))]), markup)


def process_page(page_img, page_index, eng, gpu_pool, spec, model, verify, args, priors,
                 vl_fn=None):
    """Layout, VL transcription, optional numeric verification, for one page.

    `vl_fn(image, prompt, max_tokens) -> VLReply` abstracts the transport, because
    no single one serves all three families: DeepSeek-OCR-2 cannot run through the
    HTTP server at all (see vl_client.py).
    """
    from paddleocr import LayoutDetection

    if vl_fn is None:
        def vl_fn(image, prompt, max_tokens):
            return vl_read(image, "page", max_tokens, args.block_timeout, True,
                           prompt, model, spec.strip_patterns)

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
    page.page_markdown = ""          # set in page mode

    def crop(box, pad=0):
        x1, y1, x2, y2 = box
        return page_img[max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad)]

    for b in blocks:
        if b.label in IMAGE_LABELS:
            b.status, b.note = "image", "figure region, not transcribed"
        elif b.label in QUARANTINE_LABELS:
            b.status, b.note = "quarantined", f"{b.label}: never transcribed automatically"

    # ---- VL transcription --------------------------------------------------
    futures, page_future = {}, None
    if spec.granularity == "page":
        page_future = gpu_pool.submit(vl_fn, page_img, spec.page_prompt,
                                      spec.max_tokens_page)
    else:
        for b in blocks:
            if b.status in ("image", "quarantined"):
                continue
            futures[b.index] = gpu_pool.submit(
                vl_fn, crop(b.box), block_prompt(spec, b.label),
                spec.max_tokens_page)

    # ---- CPU lane ----------------------------------------------------------
    numeric_lines: list[dict] = []
    seq = [0]

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
                    continue
                pad = args.pad
                py1, py2 = max(0, ly1 - pad), min(block_img.shape[0], ly2 + pad)
                px1, px2 = max(0, lx1 - pad), min(block_img.shape[1], lx2 + pad)
                padded = block_img[py1:py2, px1:px2]
                readings = [Reading("ppocr", numeric.keys(p_text), p_score, p_text)]
                for vname, vimg in preprocess_variants(padded):
                    if vname not in args.variants:
                        continue
                    vp, vs = eng.rec_line(vimg)
                    readings.append(Reading(f"ppocr:{vname}", numeric.keys(vp), vs, vp))
                    vt, vc = eng.tess_line(vimg, args.lang)
                    readings.append(Reading(f"tesseract:{vname}", numeric.keys(vt),
                                            (sum(vc) / len(vc)) if vc else 0.0, vt))
                t_text, t_confs = eng.tess_line(line_img, args.lang)
                readings.append(Reading("tesseract", numeric.keys(t_text),
                                        (sum(t_confs) / len(t_confs)) if t_confs else 0.0,
                                        t_text))
                seq[0] += 1
                numeric_lines.append({
                    "line_id": f"p{page_index}l{seq[0]}",
                    "block": b.index, "label": b.label,
                    "box": (b.box[0] + lx1, b.box[1] + ly1, b.box[0] + lx2, b.box[1] + ly2),
                    "readings": readings, "ppocr_text": p_text, "tesseract_text": t_text,
                    "vl_future": gpu_pool.submit(vl_fn, line_img, spec.line_prompt,
                                                 spec.max_tokens_line),
                })

    if verify:
        t = threading.Thread(target=cpu_lane, name="cpu-lane")
        t.start()
        t.join()

    # ---- collect VL --------------------------------------------------------
    def account(reply):
        page.vl_calls += 1
        page.vl_prompt_tokens += reply.prompt_tokens
        page.vl_completion_tokens += reply.completion_tokens

    if page_future is not None:
        try:
            reply = page_future.result(timeout=args.block_timeout)
            account(reply)
            page.page_markdown = reply.text
            for b in blocks:
                if b.status == "pending":
                    b.status = "ok" if reply.text else "empty"
                    b.note = "" if reply.text else "VL returned no text for the page"
            if reply.truncated:
                for b in blocks:
                    if b.status == "ok":
                        b.status, b.note = "truncated", "VL page reply hit max_tokens"
        except VLError as exc:
            for b in blocks:
                if b.status == "pending":
                    b.status, b.note = "failed", f"VL contract violation: {exc}"
        except Exception as exc:
            for b in blocks:
                if b.status == "pending":
                    b.status, b.note = "failed", f"VL page call failed: {exc}"
    else:
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
            account(reply)
            if reply.truncated:
                b.status, b.note, b.text = "truncated", "VL reply hit max_tokens", reply.text
                continue
            b.text, b.confidence = reply.text.strip(), reply.confidence
            b.status = "ok" if b.text else "empty"
            if not b.text:
                b.note = "VL returned no text for this region"

    # ---- pure-VL modes: record what the model read, marked unverified --------
    # Without this a no-vote mode reports zero numbers, which would make
    # cross-mode comparison impossible and would overstate the HTML's silence.
    if not verify:
        sources = ([("page", page.page_markdown)] if spec.granularity == "page"
                   else [(b.index, b.text) for b in blocks if b.status == "ok"])
        for where, text in sources:
            for i, q in enumerate(numeric.extract(text or "")):
                page.numbers.append({
                    "value": q.key, "display": numeric.format_key(q.key),
                    "verified": False, "unanimous": None,
                    "n_families": 0, "margin_frac": 0.0,
                    "index_in_line": i, "block": where if where != "page" else -1,
                    "block_label": "page" if where == "page" else blocks[where].label,
                    "page": page_index, "raw": q.raw, "kind": q.kind, "unit": q.unit,
                    "numeric_flags": q.flags,
                    "readings": {"vl": {"value": q.key, "confidence": 0.0}},
                    "needs_review": bool(q.flags),
                })

    # ---- fusion ------------------------------------------------------------
    for nl in numeric_lines:
        vl_conf, vl_text, vl_conf_available = {}, "", True
        try:
            reply = nl["vl_future"].result(timeout=args.block_timeout)
            account(reply)
            vl_conf_available = reply.confidence_available
            # Only build per-key confidences when the backend actually reported
            # logprobs. Otherwise the VL reading votes with a declared prior, and
            # the audit says so.
            vl_conf = vl_confidence_by_key(reply.tokens) if vl_conf_available else {}
            vl_text = reply.text
            nl["readings"].append(Reading("vl", numeric.keys(vl_text),
                                          reply.confidence, vl_text))
        except Exception as exc:
            nl["vl_error"] = str(exc)

        readings = [r for r in nl["readings"] if r.keys]
        n, kept, dropped, notes = reconcile(readings, args.min_families)
        line_suspect = bool(numeric.lost_separator_spans(nl["ppocr_text"])
                            or numeric.lost_separator_spans(nl["tesseract_text"])
                            or numeric.lost_separator_spans(vl_text))
        for k in range(n):
            per_source, seen = {}, set()
            for r in kept:
                if k >= len(r.keys):
                    continue
                key = r.keys[k]
                c = vl_conf.get(key, r.conf) if r.source == "vl" else r.conf
                per_source[r.source] = (key, c)
                seen.add(key)
            for r in dropped:
                if k < len(r.keys):
                    seen.add(r.keys[k])
            res = vote(per_source, priors, seen)
            if res is None:
                continue
            if not vl_conf_available and "vl" in per_source:
                # The backend reported no logprobs, so the VL reading has no
                # measured confidence -- it votes with a declared prior. Counting
                # it as a corroborating family would let an unmeasurable reading
                # satisfy the two-family rule, which is how an earlier version of
                # this code let a missing signal act like a strong one.
                measured = {family(e) for e in per_source if e != "vl"}
                res["n_families_measured"] = len(measured)
                res["n_families"] = len(measured)
            flags = sorted({f for q in numeric.extract(nl["ppocr_text"]) for f in q.flags})
            res.update({
                "display": numeric.format_key(res["value"]),
                "line_id": nl["line_id"], "index_in_line": k,
                "block": nl["block"], "block_label": nl["label"],
                "box": nl["box"], "page": page_index,
                "reconcile_notes": notes,
                "suspect_lost_separator": line_suspect,
                "vl_confidence_measured": vl_conf_available,
                "numeric_flags": flags,
                "dissent": [{"source": r.source, "keys": r.keys,
                             "confidence": round(r.conf, 4)} for r in dropped],
                "credible_dissent": [{"source": r.source, "keys": r.keys,
                                      "confidence": round(r.conf, 4)}
                                     for r in dropped if r.conf >= args.dissent_floor],
                "raw_text": {"vl": vl_text, "ppocr": nl["ppocr_text"],
                             "tesseract": nl["tesseract_text"]},
                "needs_review": bool(
                    line_suspect or flags or notes
                    or res["n_families"] < 2
                    or res["margin_frac"] < args.min_margin
                    or any(r.conf >= args.dissent_floor for r in dropped)
                    # An unmeasured VL confidence is not evidence of agreement.
                    or (not vl_conf_available and len(per_source) < 3)),
            })
            page.numbers.append(res)
    return page


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def gap_html(b: Block) -> str:
    word = {"image": "Figure not transcribed",
            "quarantined": "NOT TRANSCRIBED",
            "failed": "TRANSCRIPTION FAILED",
            "truncated": "TRUNCATED — CONTENT MISSING",
            "empty": "NO TEXT RECOGNISED"}[b.status]
    cls = "legend" if b.status == "image" else "gap"
    return (f'<div class="{cls}">{word}: {html.escape(b.label)} at {b.box} — '
            f'{html.escape(b.note or "see source image")}</div>')


def render_page_html(page: PageResult, spec) -> tuple[str, tuple]:
    annotate, tally = make_annotator(page.numbers)
    parts = []
    if spec.granularity == "page":
        if page.page_markdown:
            markup, slots = md_tables_to_html(page.page_markdown)
            parts.append(fill_placeholders(markup, slots, annotate))
        for b in page.blocks:
            if b.status in ("image", "quarantined", "failed", "truncated", "empty"):
                parts.append(gap_html(b))
    else:
        for b in page.blocks:
            parts.append(hybrid_ocr.render_block(b, annotate))
    return "\n".join(parts), tally()


def render_document(pages, title: str, spec, model: str, verify: bool, mode: str) -> str:
    body, total, review, gaps = [], 0, 0, 0
    for pg in pages:
        chunk, (used, avail) = render_page_html(pg, spec)
        total += len(pg.numbers)
        review += sum(1 for n in pg.numbers if n.get("needs_review"))
        gaps += sum(1 for b in pg.blocks
                    if b.status in ("failed", "truncated", "quarantined"))
        note = ""
        if used != avail:
            note = (f'<div class="banner">Provenance alignment incomplete: {used} of '
                    f'{avail} verified numbers matched the transcribed text. The rest '
                    f'are in the audit file.</div>')
        body.append(f'<section class="page" id="page-{pg.index + 1}">'
                    f'<h2>Page {pg.index + 1}</h2>{note}{chunk}</section>')

    if verify:
        unmeasured = sum(1 for p in pages for n in p.numbers
                         if n.get("vl_confidence_measured") is False)
        caveat = ""
        if unmeasured:
            caveat = (f" For {unmeasured} of them the VL backend reported no "
                      f"per-token confidence, so that reading voted with a "
                      f"declared prior rather than a measured one and does not "
                      f"count towards corroboration.")
        verified = (f"<strong>{total}</strong> numbers were read by more than one "
                    f"engine family and resolved by confidence-weighted vote; "
                    f"<strong>{review}</strong> need human review.{caveat}")
    else:
        verified = (f"<strong>No numeric verification was performed in this mode.</strong> "
                    f"All <strong>{total}</strong> numbers below are single readings "
                    f"from one model with no cross-check, and are shown dashed for "
                    f"that reason. Run a <code>+ocr</code> mode to have them voted on.")
    banner = (f'<div class="banner">Unreviewed machine transcription — mode '
              f'<code>{html.escape(mode)}</code>. {review} number(s) flagged, '
              f'{gaps} region(s) not transcribed. Do not treat as a verified record.</div>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — {html.escape(mode)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="legend">Mode <code>{html.escape(mode)}</code> · VL model
<code>{html.escape(model)}</code> · {html.escape(spec.family)} ·
{html.escape(spec.granularity)}-granularity</p>
{banner}
{"".join(body)}
<div class="legend">
<p>{verified}</p>
<p>Numbers marked <span class="num review">like this ⚑</span> need review.
<span class="num disputed">Red</span> means engines disagreed or a decimal
separator may have been lost. <span class="num unverified">Dashed</span> means the
number was never verified by a vote. The flag glyph and border weight carry the
same meaning as the colour.</p>
<p>Prose is transcribed by the VL model alone and is not verified in any mode.</p>
</div></body></html>"""
