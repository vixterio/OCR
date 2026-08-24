"""PDF to page images, for scanned documents.

The important property of a *scanned* PDF is that the capture DPI is a ceiling.
Re-rendering a 150 DPI scan at 300 DPI costs four times the pixels and adds no
information, so this module reads what is actually there:

  * if a page carries a text layer, say so -- the caller decides whether to trust
    someone else's OCR, and for clinical records the safe default is not to
  * if a page is a single full-page image, extract it at its **native** resolution
  * otherwise render, at a DPI derived from the page's own images where possible

pypdfium2 is used because it ships manylinux x86_64 and aarch64 wheels as well as
macOS ones, so this works on the deployment target and not only on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PdfPage:
    index: int                 # 0-based
    image: np.ndarray          # BGR, as cv2 expects
    dpi: float | None          # measured where known, else the render DPI used
    source: str                # 'embedded-native' | 'rendered'
    text_layer_chars: int      # 0 means a pure scan
    native_size: tuple[int, int] | None = None


def _to_bgr(pil):
    import cv2
    arr = np.array(pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def page_count(path: str) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def load_pages(path: str, render_dpi: float = 300.0, max_pages: int | None = None):
    """Yield PdfPage for each page, lazily so a 200-page record does not need to
    fit in memory at once."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        n = len(pdf)
        if max_pages:
            n = min(n, max_pages)
        for i in range(n):
            page = pdf[i]
            try:
                chars = 0
                try:
                    tp = page.get_textpage()
                    chars = len(tp.get_text_range() or "")
                except Exception:
                    chars = 0

                # A page that is one image and nothing else is a scan; take it at
                # native resolution rather than resampling it.
                images = [o for o in page.get_objects() if getattr(o, "type", None) == 3]
                if len(images) == 1 and chars == 0:
                    obj = images[0]
                    meta = None
                    try:
                        meta = obj.get_metadata()
                    except Exception:
                        meta = None
                    try:
                        pil = obj.get_bitmap().to_pil()
                        dpi = float(getattr(meta, "horizontal_dpi", 0) or 0) or None
                        yield PdfPage(i, _to_bgr(pil), dpi, "embedded-native", chars,
                                      (pil.width, pil.height))
                        continue
                    except Exception:
                        pass  # fall through to rendering

                scale = render_dpi / 72.0
                pil = page.render(scale=scale).to_pil()
                yield PdfPage(i, _to_bgr(pil), render_dpi, "rendered", chars,
                              (pil.width, pil.height))
            finally:
                page.close()
    finally:
        pdf.close()


def describe(p: PdfPage) -> str:
    h, w = p.image.shape[:2]
    dpi = f"{p.dpi:.0f} dpi" if p.dpi else "dpi unknown"
    layer = "scan" if p.text_layer_chars == 0 else f"{p.text_layer_chars} chars of text layer"
    return f"page {p.index + 1}: {w}x{h}px, {dpi}, {p.source}, {layer}"
