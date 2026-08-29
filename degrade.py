"""Degrade a clean fixture the way a real scanner does, keeping ground truth.

The synthetic fixtures are the easiest input these modes will ever see, and it
shows: seven of eleven configurations score 100% recall and 100% precision on
them, so the fixtures can no longer tell the modes apart. A benchmark everything
passes measures nothing.

The fix is not a harder document -- that would need new ground truth, and hand
labelling is where benchmarks acquire their errors. It is the same document,
photocopied badly. Every transform here is a pure function of the clean raster,
so the set of numbers on the page is unchanged by construction and the existing
labels in evaluate.py stay correct. What changes is only how hard they are to
read.

The five axes are the five things that actually go wrong with scanned clinical
paper, in rough order of how often they do:

  resolution  the page was scanned at 150 DPI, or faxed at 100. Thin strokes
              and the gap inside a comma fall below one pixel and stop existing.
  jpeg        almost every scanner emits JPEG. Ringing around a glyph edge is
              indistinguishable from a real mark at low quality.
  skew        paper fed by hand is never square. Costs line-based engines more
              than page-based ones, which is the point of measuring it.
  noise       photocopier speckle and sensor grain. Isolated dark pixels look
              like decimal points, which is the failure that matters here.
  blur        out of focus, or a copy of a copy. Merges adjacent glyphs.

    .venv/bin/python degrade.py euro_table_sample.png --level medium
    .venv/bin/python degrade.py euro_table_sample.png --all
    .venv/bin/python degrade.py euro_table_sample.png --axis jpeg --amount 25

Noise is seeded from the output filename, so a given fixture is byte-identical
every time it is built. Comparing a mode against last week's number is only
meaningful if the input did not move.
"""
from __future__ import annotations

import argparse
import io
import os
import zlib

import numpy as np
from PIL import Image, ImageFilter

# Severity ladder. Each level is all five axes at once, because real degradation
# arrives that way -- a faxed photocopy is low resolution AND noisy AND skewed,
# and the interactions are the interesting part. Isolated axes are available via
# --axis when a failure needs attributing to a cause.
LEVELS = {
    #        scale  jpeg  skew°  noise  blur   grey
    "light":  (0.75,  75,  0.4,   4,    0.0,  255),
    "medium": (0.55,  45,  1.2,   9,    0.4,  247),
    "heavy":  (0.40,  25,  2.0,  16,    0.8,  238),
}


def _rng(seed_text: str) -> np.random.Generator:
    """A generator seeded by name, so each fixture is reproducible and distinct."""
    return np.random.default_rng(zlib.crc32(seed_text.encode()))


def resample(img: Image.Image, scale: float) -> Image.Image:
    """Scan at a lower DPI, then view at the original size.

    Down with LANCZOS and back up with BICUBIC. The round trip is the point: the
    information destroyed going down does not come back, but the image stays the
    size the ground truth expects, so nothing downstream has to know.
    """
    if scale >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return small.resize((w, h), Image.BICUBIC)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Round trip through JPEG at a given quality."""
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def skew(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate, expanding the canvas so no glyph is ever cut off.

    expand=True is not a detail. The last fixture bug was a clipped descender
    that made every engine look wrong for a week; a rotation that crops corners
    would reintroduce exactly that, and it would again look like an OCR failure
    rather than a fixture failure.
    """
    if not degrees:
        return img
    return img.rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def speckle(img: Image.Image, sigma: float, seed: str) -> Image.Image:
    """Gaussian sensor grain plus a sparse scatter of hard dark pixels.

    The Gaussian part is what a cheap sensor does. The scatter is what a dirty
    platen does, and it is the more dangerous of the two here: a lone dark pixel
    next to a digit is a decimal point, and a decimal point invented in a dose is
    a hundredfold error that reads as perfectly plausible.
    """
    if sigma <= 0:
        return img
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    r = _rng(seed)
    a += r.normal(0, sigma, a.shape)
    dirt = r.random(a.shape[:2]) < (sigma / 4000.0)
    a[dirt] = r.integers(0, 90, (int(dirt.sum()), 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def blur(img: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def greyed(img: Image.Image, white: int) -> Image.Image:
    """Lift the black point and drop the white point: a copy of a copy.

    Compresses the dynamic range rather than shifting it, so adaptive
    thresholding still has something to work with but a fixed threshold does not.
    """
    if white >= 255:
        return img
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    black = (255 - white) * 0.9
    return Image.fromarray(np.clip(black + a * (white - black) / 255.0, 0, 255).astype(np.uint8))


def apply(img: Image.Image, scale, quality, degrees, sigma, radius, white, seed):
    """The pipeline, in the order a scanner actually applies it.

    Order matters and is not arbitrary: the page is skewed on the platen before
    it is sampled, sampling happens before the optics blur, grain is added by the
    sensor, and JPEG is the last thing the firmware does. Compressing before
    adding noise would let JPEG smooth away the very speckle being tested.
    """
    img = skew(img, degrees)
    img = resample(img, scale)
    img = blur(img, radius)
    img = greyed(img, white)
    img = speckle(img, sigma, seed)
    return jpeg(img, quality)


def build(src: str, out: str, params) -> str:
    img = Image.open(src).convert("RGB")
    apply(img, *params, seed=os.path.basename(out)).save(out)
    print(f"{out}  {Image.open(out).size}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("src", help="clean fixture PNG")
    ap.add_argument("--level", choices=sorted(LEVELS), help="all five axes at once")
    ap.add_argument("--all", action="store_true", help="build every level")
    ap.add_argument("--axis", choices=["scale", "jpeg", "skew", "noise", "blur", "grey"],
                    help="one axis alone, to attribute a failure to a cause")
    ap.add_argument("--amount", type=float, help="value for --axis")
    ap.add_argument("--outdir", default="fixtures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0].replace("_sample", "")

    if args.axis:
        if args.amount is None:
            ap.error("--axis needs --amount")
        base = {"scale": 1.0, "jpeg": 100, "skew": 0.0, "noise": 0.0, "blur": 0.0, "grey": 255}
        base[args.axis] = args.amount
        p = (base["scale"], int(base["jpeg"]), base["skew"],
             base["noise"], base["blur"], int(base["grey"]))
        amt = f"{args.amount:g}".replace(".", "p")
        build(args.src, f"{args.outdir}/{stem}_{args.axis}{amt}.png", p)
        return

    levels = sorted(LEVELS) if args.all else [args.level or "medium"]
    for lv in levels:
        build(args.src, f"{args.outdir}/{stem}_{lv}.png", LEVELS[lv])


if __name__ == "__main__":
    main()
