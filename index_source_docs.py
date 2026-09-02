"""Index the generator's own documents: one PDF, one HTML, guaranteed to match.

This exists because the obvious pairing was wrong. The merged_bundles corpus and
the Downloads/output corpus are separate generation runs: they share patient
names and case ids, but the documents differ. Measured on six documents, the
number overlap between an output/ HTML and the corresponding BUNDLE page was
~50% -- while the overlap between that same HTML and output/'s OWN pdf was 100%.
Fifty percent is what two different documents about one patient share by
coincidence: dates, reference ranges, small integers. Comparing across the two
corpora would have reported a catastrophic OCR failure that was entirely an
artefact of grading against the wrong paper.

So the comparison uses output/'s own PDFs. They are single documents rather than
shuffled bundles, which loses the page-interleaving realism but gains an exact
per-document ground truth -- and the format question the comparison exists to
answer does not depend on shuffling.

    .venv/bin/python index_source_docs.py --cases 001 002 003 \
        --types hausarztbrief laborbefund --outdir work/r1
"""
from __future__ import annotations

import argparse
import json
import os

SRC = "/Users/vixterio/Downloads/output"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import pypdfium2 as pdfium
    os.makedirs(args.outdir, exist_ok=True)
    folders = {d.split("_")[1]: d for d in os.listdir(SRC) if d.startswith("case_")}
    index, pages = [], 0
    for cid in args.cases:
        folder = folders.get(cid)
        if not folder:
            print(f"  case {cid}: no folder"); continue
        base = os.path.join(SRC, folder)
        plan = json.load(open(os.path.join(base, "case_plan.json"), encoding="utf-8"))
        for f in sorted(os.listdir(base)):
            if not f.endswith(".pdf"):
                continue
            stem = f[:-4]
            dtype = stem.split("_", 1)[1] if "_" in stem else stem
            if args.types and dtype not in args.types:
                continue
            html = os.path.join(base, f"{stem}.html")
            if not os.path.exists(html):
                print(f"  {stem}: no matching HTML, skipped"); continue
            pdf = os.path.join(base, f)
            doc = pdfium.PdfDocument(pdf); n = len(doc); doc.close()
            index.append({"pdf": pdf, "source_html": html, "case_id": cid,
                          "patient": plan["patient_name"], "document_id": stem,
                          "document_type": dtype, "n_pages": n})
            pages += n
            print(f"  {plan['patient_name']:22} {stem:30} {n} pages")
    if not index:
        raise SystemExit("nothing indexed")
    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    print(f"\n{len(index)} documents, {pages} pages -> {args.outdir}/index.json")


if __name__ == "__main__":
    main()
