"""Pull one patient's documents out of a shuffled bundle, as separate PDFs.

The bundles interleave pages from many documents and many patients, which is
realistic but makes comparison against the source impossible: the generator
produced one HTML file per document, and a bundle-wide transcript has no
boundary to line up against it. This reassembles a document from the pages the
manifest says belong to it, in order, as its own PDF -- so one OCR run produces
one HTML that can be diffed against exactly one source file.

Pages are copied, not re-rendered, so the extracted PDF is bit-identical page
content to the bundle. Whatever the OCR sees here it would have seen there.

    .venv/bin/python extract_case.py bundle_053.pdf --cases 034 024 017 \
        --types laborbefund medikationsplan hausarztbrief --outdir work/round1

Writes an index.json alongside, mapping each extracted PDF to the source HTML it
should be compared with, so the scorer never has to guess the pairing.
"""
from __future__ import annotations

import argparse
import json
import os

BUNDLE_DIR = "merged_bundles"
SOURCE_DIR = "/Users/vixterio/Downloads/output"


def source_dir_for(case_id: str) -> str | None:
    """The generator's folder for a case, found by its numeric prefix."""
    if not os.path.isdir(SOURCE_DIR):
        return None
    for name in sorted(os.listdir(SOURCE_DIR)):
        if name.startswith(f"case_{case_id}_"):
            return os.path.join(SOURCE_DIR, name)
    return None


def _norm(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z]", "",
                  "".join(c for c in s if not unicodedata.combining(c)).lower())


def source_html_for(case_id: str, document_id: str, patient: str) -> str | None:
    """The HTML this document was rendered from, or None if it is a different patient.

    The case_id alone is NOT sufficient and must never be trusted on its own. The
    source corpus and the bundle corpus are separate generation runs whose case
    ids collide numerically: 45 of the 50 source cases carry a different patient
    under the same id, so pairing on the id silently compares one person's
    transcript against another person's document. Only 5 cases genuinely
    correspond, and this checks the name before returning a path.
    """
    d = source_dir_for(case_id)
    if not d:
        return None
    plan = os.path.join(d, "case_plan.json")
    if os.path.exists(plan):
        with open(plan, encoding="utf-8") as fh:
            if _norm(json.load(fh).get("patient_name", "")) != _norm(patient):
                return None
    path = os.path.join(d, f"{document_id}.html")
    return path if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--types", nargs="*", default=None,
                    help="document types to extract; default is all of them")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import pypdfium2 as pdfium
    man = {b["bundle"]: b for b in json.load(open(f"{BUNDLE_DIR}/manifest.json"))}
    entry = man.get(args.bundle)
    if entry is None:
        raise SystemExit(f"{args.bundle} is not in the manifest")

    os.makedirs(args.outdir, exist_ok=True)
    src = pdfium.PdfDocument(os.path.join(BUNDLE_DIR, args.bundle))
    index, pages_total = [], 0
    try:
        for doc in entry["documents"]:
            if doc["case_id"] not in args.cases:
                continue
            if args.types and doc["document_type"] not in args.types:
                continue
            pages = doc["pages_in_bundle"]
            out_pdf = os.path.join(
                args.outdir, f"case{doc['case_id']}_{doc['document_id']}.pdf")
            new = pdfium.PdfDocument.new()
            # pages_in_bundle is 1-based in the manifest; import_pages is 0-based.
            new.import_pages(src, [p - 1 for p in pages])
            new.save(out_pdf)
            new.close()
            html = source_html_for(doc["case_id"], doc["document_id"],
                                   doc["patient_name"])
            index.append({
                "pdf": out_pdf, "source_html": html,
                "case_id": doc["case_id"], "patient": doc["patient_name"],
                "document_id": doc["document_id"],
                "document_type": doc["document_type"],
                "institution": doc.get("institution"),
                "bundle": args.bundle, "pages_in_bundle": pages,
            })
            pages_total += len(pages)
            mark = "" if html else "   [no source HTML]"
            print(f"  {os.path.basename(out_pdf):44} {len(pages)} pages  "
                  f"{doc['patient_name']}{mark}")
    finally:
        src.close()

    if not index:
        raise SystemExit("nothing matched those cases/types")
    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    missing = sum(1 for r in index if not r["source_html"])
    print(f"\n{len(index)} documents, {pages_total} pages -> {args.outdir}")
    if missing:
        print(f"WARNING: {missing} have no source HTML and cannot be compared")


if __name__ == "__main__":
    main()
