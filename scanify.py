"""Turn a born-digital PDF into something shaped like a scan of itself.

The bundles are born-digital: their pages are drawn from font instructions, so
rendering them gives perfectly sharp glyphs, perfectly square baselines, pure
black on pure white, no noise and no compression. The models are already doing
real OCR on them -- the text layer is never fed to any model, only counted -- but
they are doing it on the easiest input a page can present. A 96% word score on
that is not a prediction of what a real scanned referral will do.

This renders each page, puts it through the same degradations as degrade.py, and
writes the result back as an image-only PDF. That last part matters: the output
has no text layer at all, so pdf_input sees a single image and zero characters
and takes its genuine scan path, exactly as it would for paper that went through
a feeder. Nothing downstream needs to know the difference.

Ground truth is unaffected. score_bundle.py reads the text layer of the ORIGINAL
bundle, and the manifests are keyed by filename, so a scanned copy is scored
against the same answers as the digital one. That is the whole point: the only
variable that moves is how hard the pixels are to read.

    .venv/bin/python scanify.py merged_bundles/bundle_001.pdf --level medium --pages 20
    .venv/bin/python scanify.py merged_bundles/bundle_001.pdf --level heavy --dpi 200
"""
from __future__ import annotations

import argparse
import os

from PIL import Image

import degrade


def render(pdf_path: str, dpi: float, limit: int | None):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    n = len(doc) if limit is None else min(limit, len(doc))
    try:
        for i in range(n):
            page = doc[i]
            try:
                yield page.render(scale=dpi / 72.0).to_pil().convert("RGB")
            finally:
                page.close()
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--level", choices=sorted(degrade.LEVELS), default="medium")
    ap.add_argument("--dpi", type=float, default=200.0,
                    help="scanner resolution to simulate. Most clinical scanners "
                         "run at 200-300; the default is deliberately not 300, "
                         "because a page that was never scanned at 300 should not "
                         "be handed to OCR as though it had been")
    ap.add_argument("--pages", type=int, default=None)
    ap.add_argument("--quality", type=int, default=80,
                    help="JPEG quality inside the PDF, as a scanner would store it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    out = args.out or f"merged_bundles/{stem}_scan_{args.level}.pdf"
    params = degrade.LEVELS[args.level]

    pages = []
    for i, img in enumerate(render(args.pdf, args.dpi, args.pages)):
        # Seeded per page, so rebuilding the same fixture gives the same pixels
        # and a score can be compared with last week's.
        pages.append(degrade.apply(img, *params, seed=f"{stem}-{args.level}-{i}"))
        print(f"  page {i + 1}: {pages[-1].size}")
    if not pages:
        raise SystemExit("no pages rendered")

    # Image-only PDF: no fonts, no text objects, nothing to extract. This is what
    # makes it a scan rather than a picture of one.
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:],
                  resolution=args.dpi, quality=args.quality)
    mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({len(pages)} pages, {mb:.1f} MB, {args.level} at {args.dpi:.0f} dpi)")


if __name__ == "__main__":
    main()
