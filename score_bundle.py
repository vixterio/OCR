"""Score a run against a bundle's own text layer and its manifests.

The synthetic bundles carry three independent kinds of ground truth, and they
answer different questions:

  text layer   Every page of these PDFs has an embedded text layer, which is
               what the generator drew. Rendering the page and OCRing it is
               therefore a closed experiment: the exact right answer is inside
               the file, and no hand labelling is involved.
  manifest     Which documents occupy which pages, and for each one the patient
               identity a downstream system has to recover: family name, given
               name, date of birth, insurance number (KVNR).
  numbers      Extracted from the text layer with the same numeric.py the
               pipeline uses, so the number-level score here is comparable to
               the fixture scores in evaluate.py.

Five metrics, because they fail differently:

  PII          Did the four identifying fields survive? This is the one that
               decides whether a record can be filed against the right patient.
               A page can be 99% correct and still useless if the 1% was the
               insurance number. Fields not printed on the scored pages are
               excluded, not counted as failures.
  footer       Every page ends with "Patient: <name> | geb. <dob> | Dok-ID: ..."
               and on continuation pages that footer is the only place the
               patient is named. Tracked alone because a model can drop every
               footer and still score 92% on words.
  content      Word-level recall and precision against the text layer, order
               independent, because reading order is a layout choice and not an
               OCR error. Precision is what catches a model that pads its output.
  transcript   Numbers present in the prose the model wrote.
  resolved     Numbers the pipeline actually decided on, from the audit. The two
               differ exactly where the +ocr vote does its work: the prose is the
               same whether or not a vote ran, so scoring only the prose made
               granite and granite+ocr tie on every metric by construction.

    .venv/bin/python score_bundle.py output/b1_granite/audit.json
    .venv/bin/python score_bundle.py output/b1_*/audit.json --table
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numeric

BUNDLE_DIR = "merged_bundles"

# Markdown and DocTags scaffolding the models emit around the text. Stripped
# before comparison: a pipe character in a table border is not a transcription
# error, and counting it as one would punish whichever model formats most.
_MD = re.compile(r"(\|)|(^\s{0,3}#{1,6}\s)|([*_`>])|(^\s*[-:]{3,}\s*$)|(<[^>]{1,80}>)",
                 re.MULTILINE)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def norm(s: str) -> str:
    """Casefold and strip accents, so Müller and MULLER compare equal.

    Deliberate: an umlaut lost to OCR is a real defect, but it is a *different*
    defect from failing to find the patient, and the PII check below reports
    both -- exact first, then accent-insensitive -- so the two do not hide each
    other.
    """
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def words(text: str) -> Counter:
    return Counter(_WORD.findall(norm(_MD.sub(" ", text))))


def numbers_in(text: str) -> Counter:
    out = Counter()
    for k in numeric.keys(_MD.sub(" ", text)):
        body = k.split(":")[1] if ":" in k else k
        for part in body.split("/"):
            v, _ = numeric.normalise(part)
            if v is not None:
                out[v] += 1
    return out


def resolved_numbers(page: dict) -> Counter:
    """The values the pipeline actually decided on, from the audit.

    Distinct from the numbers in the transcript, and the distinction is the whole
    point of the +ocr modes. numbers_in() reads the VL model's prose, which is
    identical whether or not a vote ran; the vote's output lives in the audit's
    `numbers` array. Scoring only the prose made granite and granite+ocr tie on
    every metric by construction -- the vote could have corrected or ruined every
    figure on the page and the score would not have moved.
    """
    out = Counter()
    for rec in page.get("numbers", []):
        key = rec.get("value") or ""
        body = key.split(":")[1] if key.count(":") >= 1 else key
        for part in body.split("/"):
            v, _ = numeric.normalise(part)
            if v is not None:
                out[v] += 1
    return out


def page_text(page: dict) -> str:
    """Everything a mode transcribed for one page.

    Page-granularity models (DeepSeek, Qwen, Granite) fill page_markdown;
    block-granularity PaddleOCR-VL fills per-block text and leaves it empty.
    Both are joined so the two families are scored on the same basis.
    """
    parts = [page.get("page_markdown") or ""]
    parts += [b.get("text") or "" for b in page.get("blocks", [])]
    return "\n".join(p for p in parts if p)


def truth_pages(pdf_path: str, n: int) -> list[str]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    out = []
    try:
        for i in range(min(n, len(doc))):
            pg = doc[i]
            try:
                out.append(pg.get_textpage().get_text_range() or "")
            finally:
                pg.close()
    finally:
        doc.close()
    return out


def kvnr_variants(k: str) -> list[str]:
    """A KVNR with and without its grouping spaces."""
    bare = re.sub(r"\s+", "", k)
    return [k, bare]


def dob_variants(iso: str) -> list[str]:
    """The same date as the generator writes it, and as ISO."""
    y, m, d = iso.split("-")
    return [f"{d}.{m}.{y}", iso, f"{d}.{m}.{y[2:]}"]


def pii_expectations(bundle: str, n_pages: int):
    """Documents whose pages fall inside the scored range, with their identity."""
    man = {b["bundle"]: b for b in json.load(open(f"{BUNDLE_DIR}/manifest.json"))}
    bulk = {s["case_id"]: s for s in
            json.load(open(f"{BUNDLE_DIR}/bulk_upload_manifest.json"))}
    entry = man.get(bundle)
    if entry is None:
        return []
    out = []
    for doc in entry["documents"]:
        pages = [p for p in doc["pages_in_bundle"] if p <= n_pages]
        if not pages:
            continue
        s = bulk.get(doc["case_id"])
        if not s:
            continue
        pii = s["expected_pii"]
        out.append({"case": doc["case_id"], "type": doc["document_type"],
                    "pages": pages, "scenario": s["scenario_type"],
                    "family": pii["family"], "given": pii["given"],
                    "dob": pii["dob_iso"], "kvnr": pii["kvnr"]})
    return out


_SCAN_SUFFIX = re.compile(r"_scan_[a-z]+$")


def origin_of(bundle: str) -> str:
    """The born-digital bundle a scanned copy was made from.

    scanify.py writes an image-only PDF, which is what makes it a fair scan
    test -- and also means it carries no text layer to be scored against. Both
    the ground truth and the manifest lookup therefore have to come from the
    original. The degradation is a pure image transform of the same render, so
    the words, numbers and patients on the page are unchanged by construction
    and the original's answers remain the right ones.
    """
    stem, ext = os.path.splitext(bundle)
    return _SCAN_SUFFIX.sub("", stem) + ext


def score(audit_path: str):
    d = json.load(open(audit_path, encoding="utf-8"))
    src = (d.get("input") or "")
    scanned = os.path.basename(src)
    bundle = origin_of(scanned)
    pdf = os.path.join(BUNDLE_DIR, bundle)
    if not os.path.exists(pdf):
        pdf = src if os.path.exists(src) else pdf
    if not os.path.exists(pdf):
        return None
    pages = d.get("pages", [])
    truth = truth_pages(pdf, len(pages))

    # Footer recovery, tracked separately because it decides patient identity.
    # Every page of these bundles carries a footer of the form
    #   "Patient: <name> | geb. <dob> | Dok-ID: <case>-<doc> Seite n von m"
    # and on continuation pages that footer is the ONLY place the patient is
    # named. A model that treats footers as page furniture and drops them can
    # score 94% on words and still lose the identity of the record, which is the
    # failure that misfiles it. Aggregate accuracy hides this completely.
    foot_exp = foot_hit = 0
    w_hit = w_exp = w_found = 0
    n_exp = n_found = n_hit = 0
    r_found = r_hit = 0
    per_page = []
    for i, page in enumerate(pages):
        t = words(truth[i]) if i < len(truth) else Counter()
        o = words(page_text(page))
        hit = sum((t & o).values())
        w_hit += hit; w_exp += sum(t.values()); w_found += sum(o.values())
        tn = numbers_in(truth[i]) if i < len(truth) else Counter()
        on = numbers_in(page_text(page))
        nh = sum((tn & on).values())
        n_hit += nh; n_exp += sum(tn.values()); n_found += sum(on.values())
        rn = resolved_numbers(page)
        r_hit += sum((tn & rn).values()); r_found += sum(rn.values())
        if i < len(truth) and "Dok-ID" in truth[i]:
            foot_exp += 1
            foot_hit += "Dok-ID" in page_text(page)
        per_page.append({"page": i + 1, "w_recall": hit / max(1, sum(t.values()))})

    # PII: search the pages a document actually occupies.
    text_by_page = {i + 1: norm(page_text(p)) for i, p in enumerate(pages)}
    truth_by_page = {i + 1: norm(t) for i, t in enumerate(truth)}
    exp = pii_expectations(bundle, len(pages))
    fields = Counter(); found = Counter()
    misses = []
    unreachable = Counter()

    def present(cands, blob):
        bare = re.sub(r"\s+", "", blob)
        return any(norm(c) in blob or norm(re.sub(r"\s+", "", c)) in bare
                   for c in cands)

    for doc in exp:
        blob = " ".join(text_by_page.get(p, "") for p in doc["pages"])
        src = " ".join(truth_by_page.get(p, "") for p in doc["pages"])
        for field, cands in (("family", [doc["family"]]), ("given", [doc["given"]]),
                             ("dob", dob_variants(doc["dob"])),
                             ("kvnr", kvnr_variants(doc["kvnr"]))):
            # Only ask for what is actually printed on the pages being scored.
            # A document's manifest entry gives the patient's full identity, but
            # the insurance number is typically printed once, on the document's
            # first page. Demanding it from a continuation page counts a field
            # that was never there as an OCR failure -- which made granite look
            # like it lost 3 KVNRs when 2 of them are not in the source at all.
            if not present(cands, src):
                unreachable[field] += 1
                continue
            fields[field] += 1
            if present(cands, blob):
                found[field] += 1
            else:
                misses.append((doc["case"], doc["scenario"], field,
                               cands[0], doc["pages"]))

    toks = sum(p.get("vl_prompt_tokens", 0) + p.get("vl_completion_tokens", 0)
               for p in pages)
    npages = max(1, len(pages))
    return {
        "path": audit_path, "bundle": bundle, "scanned": scanned != bundle,
        "mode": d.get("mode", "?"), "model": (d.get("vl_model") or "").split("/")[-1],
        "line_reads": d.get("vl_line_reads"), "pages": len(pages),
        "w_recall": w_hit / max(1, w_exp), "w_precision": w_hit / max(1, w_found),
        "n_recall": n_hit / max(1, n_exp), "n_precision": n_hit / max(1, n_found),
        "n_missed": n_exp - n_hit, "n_spurious": n_found - n_hit,
        "r_recall": r_hit / max(1, n_exp), "r_precision": r_hit / max(1, r_found),
        "r_missed": n_exp - r_hit, "r_spurious": r_found - r_hit,
        "pii_fields": dict(fields), "pii_found": dict(found), "pii_misses": misses,
        "pii_unreachable": dict(unreachable),
        "pii_rate": sum(found.values()) / max(1, sum(fields.values())),
        "footer_rate": foot_hit / max(1, foot_exp), "footer_exp": foot_exp,
        # >1 means the model emitted more text than the page holds. Repetition
        # and invention both show up here before they show up as a wrong number.
        "length_ratio": w_found / max(1, w_exp),
        "tokens": toks, "tokens_per_page": toks / npages,
        # Throughput is the honest cost unit for a local model. There is no
        # per-token bill; what a page costs is a slice of a machine, so the
        # figure that decides capacity is how many pages one machine clears per
        # hour. The --rate flag converts tokens into what a hosted VLM would
        # charge instead, for comparison against buying the capacity.
        "pages_per_hour": 3600.0 / max(1e-9, (d.get("wall_seconds") or 0.0) / npages),
        "vl_calls": sum(p.get("vl_calls", 0) for p in pages),
        "wall": d.get("wall_seconds") or 0.0,
        "wall_per_page": (d.get("wall_seconds") or 0.0) / npages,
        "review": sum(1 for p in pages for n in p.get("numbers", [])
                      if n.get("needs_review")),
        "gaps": sum(1 for p in pages for b in p.get("blocks", [])
                    if b.get("status") in ("failed", "truncated", "quarantined")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("audits", nargs="+")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--misses", action="store_true", help="list every PII field lost")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="USD per 1M tokens, to convert tokens/page into a cost. "
                         "These models run locally and cost nothing per token, so "
                         "this is only meaningful as 'what a hosted VLM would bill'")
    args = ap.parse_args()

    paths = []
    for a in args.audits:
        paths.extend(sorted(glob.glob(a)) if "*" in a else [a])
    rows = [r for r in (score(p) for p in paths) if r]
    if not rows:
        sys.exit("nothing scored")
    rows.sort(key=lambda r: -r["pii_rate"])

    if args.table:
        cost = f"{'$/1k pg':>8}" if args.rate else ""
        print(f"{'mode':14} {'pages':>5} {'PII':>6} {'foot':>5} {'name':>5} {'dob':>5} {'kvnr':>5} "
              f"{'word R':>7} {'word P':>7} {'txt R':>6} {'txt P':>6} "
              f"{'num R':>6} {'num P':>6} "
              f"{'tok/pg':>7} {'s/pg':>6} {'pg/hr':>6}{cost}")
        print("-" * (122 + len(cost)))
        for r in rows:
            f, g = r["pii_found"], r["pii_fields"]
            nm = (f.get("family", 0) + f.get("given", 0)) / max(1, g.get("family", 0) + g.get("given", 0))
            c = (f"{r['tokens_per_page'] * args.rate / 1000:8.2f}" if args.rate else "")
            print(f"{r['mode']:14} {r['pages']:5} {r['pii_rate']*100:5.1f}% "
                  f"{r['footer_rate']*100:4.0f}% {nm*100:4.0f}% {f.get('dob',0)/max(1,g.get('dob',1))*100:4.0f}% "
                  f"{f.get('kvnr',0)/max(1,g.get('kvnr',1))*100:4.0f}% "
                  f"{r['w_recall']*100:6.1f}% {r['w_precision']*100:6.1f}% "
                  f"{r['n_recall']*100:5.1f}% {r['n_precision']*100:5.1f}% "
                  f"{r['r_recall']*100:5.1f}% {r['r_precision']*100:5.1f}% "
                  f"{r['tokens_per_page']:7.0f} {r['wall_per_page']:6.1f} "
                  f"{r['pages_per_hour']:6.0f}{c}")
    else:
        for r in rows:
            print(f"\n{r['mode']}  ({r['model']}, {r['pages']} pages of {r['bundle']})")
            un = r["pii_unreachable"]
            print(f"  PII      {r['pii_rate']*100:.1f}%  " +
                  "  ".join(f"{k} {r['pii_found'].get(k,0)}/{v}"
                            for k, v in sorted(r["pii_fields"].items())) +
                  (f"   [{sum(un.values())} field(s) not printed on these pages, "
                   f"excluded: {un}]" if un else ""))
            print(f"  content  recall {r['w_recall']*100:.1f}%  precision {r['w_precision']*100:.1f}%"
                  f"  output/source length {r['length_ratio']:.2f}x")
            print(f"  footer   {r['footer_rate']*100:.0f}% of {r['footer_exp']} pages "
                  f"(carries the patient identity on continuation pages)")
            print(f"  numbers  transcript recall {r['n_recall']*100:.1f}%  "
                  f"precision {r['n_precision']*100:.1f}%  "
                  f"missed {r['n_missed']}  spurious {r['n_spurious']}")
            print(f"           resolved   recall {r['r_recall']*100:.1f}%  "
                  f"precision {r['r_precision']*100:.1f}%  "
                  f"missed {r['r_missed']}  spurious {r['r_spurious']}")
            print(f"  cost     {r['tokens_per_page']:.0f} tokens/page, "
                  f"{r['wall_per_page']:.1f}s/page, {r['vl_calls']} VL calls, "
                  f"{r['review']} flagged, {r['gaps']} gaps")
    if args.misses:
        print("\nPII fields lost:")
        for r in rows:
            for case, scen, field, want, pages in r["pii_misses"]:
                print(f"  {r['mode']:14} case {case} p{pages} {field:6} {want!r} ({scen})")


if __name__ == "__main__":
    main()
