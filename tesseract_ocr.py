"""Tesseract OCR via pytesseract.

Unlike the PaddleOCR scripts, Tesseract needs no model download and no server —
it is a native binary that pytesseract shells out to. It is far lighter than
PaddleOCR-VL (tens of MB, no GPU) but weaker on dense or non-Latin layouts.

Install (macOS):

    brew install tesseract          # the binary
    brew install tesseract-lang     # all extra language packs (~1.5 GB)
    .venv/bin/pip install pytesseract pillow

Language packs are separate downloads; `--list` shows what you have.

For European text prefer a *script* model over chaining languages. One
`script/Latin` pass reads Polish, Czech, Hungarian, Turkish and the rest
without being told which; `-l pol+ces+hun+...` is both slower and less
accurate, because every added language widens the character set Tesseract has
to choose from.

    .venv/bin/python tesseract_ocr.py --list
    .venv/bin/python tesseract_ocr.py page.png --script latin
    .venv/bin/python tesseract_ocr.py page.png -l pol      # better, if known
"""
import argparse
import os
import shutil
import sys

import pytesseract
from PIL import Image

OUT_DIR = "output/tesseract"

# Script-level models: one model per writing system, covering every language in
# it. These are the three European scripts.
SCRIPTS = {"latin": "script/Latin", "cyrillic": "script/Cyrillic", "greek": "script/Greek"}

# European languages by script, for --list. Codes only appear here if
# tesseract-lang actually ships them.
EUROPEAN = {
    "Latin": ("eng fra deu spa ita por nld dan swe nor fin isl pol ces slk slv hrv bos "
              "srp_latn hun ron lit lav est sqi mlt gle gla cym bre cat eus glg tur ltz "
              "fao oci cos epo lat frm enm ita_old spa_old").split(),
    "Cyrillic": "rus ukr bel bul srp mkd".split(),
    "Greek": "ell grc".split(),
}


def resolve_binary():
    """pytesseract shells out to `tesseract`; fail loudly if it isn't on PATH."""
    found = shutil.which("tesseract")
    if not found:
        sys.exit(
            "tesseract binary not found on PATH.\n"
            "Install it with: brew install tesseract"
        )
    pytesseract.pytesseract.tesseract_cmd = found
    return found


def main():
    parser = argparse.ArgumentParser(description="Run Tesseract OCR on an image.")
    parser.add_argument("image", nargs="?", default="demo.png")
    parser.add_argument("-l", "--lang", default="eng", help="language code(s), e.g. eng or eng+fra")
    parser.add_argument("--script", choices=sorted(SCRIPTS),
                        help="use a whole-script model instead of named languages")
    parser.add_argument("--psm", default="3", help="page segmentation mode (3=auto, 6=block, 7=line)")
    parser.add_argument("--list", action="store_true", help="list installed languages and exit")
    args = parser.parse_args()

    binary = resolve_binary()
    print(f"tesseract {pytesseract.get_tesseract_version()} ({binary})")

    langs = pytesseract.get_languages(config="")
    if args.list:
        print(f"{len(langs)} languages installed")
        for script, codes in EUROPEAN.items():
            have = [c for c in codes if c in langs]
            missing = [c for c in codes if c not in langs]
            print(f"\n  {script} ({len(have)}/{len(codes)}): {' '.join(have)}")
            if missing:
                print(f"    missing: {' '.join(missing)}")
        print("\n  script models:", ", ".join(SCRIPTS.values()))
        return

    if args.script:
        # Script models live in a subdirectory, so get_languages() never lists
        # them; check the file instead of the language list.
        args.lang = SCRIPTS[args.script]
    else:
        for code in args.lang.split("+"):
            if code not in langs:
                sys.exit(
                    f"language {code!r} is not installed.\n"
                    "Run --list to see what you have, or install more with:\n"
                    "  brew install tesseract-lang"
                )

    image = Image.open(args.image)
    config = f"--psm {args.psm}"

    text = pytesseract.image_to_string(image, lang=args.lang, config=config)
    # TSV carries per-word boxes and confidences, which plain text throws away.
    tsv = pytesseract.image_to_data(image, lang=args.lang, config=config)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    txt_path = os.path.join(OUT_DIR, f"{stem}.txt")
    tsv_path = os.path.join(OUT_DIR, f"{stem}.tsv")
    with open(txt_path, "w") as fh:
        fh.write(text)
    with open(tsv_path, "w") as fh:
        fh.write(tsv)

    words = [
        line.split("\t")
        for line in tsv.splitlines()[1:]
        if len(line.split("\t")) == 12 and line.split("\t")[11].strip()
    ]
    confs = [float(w[10]) for w in words if w[10] not in ("-1", "")]
    print(f"{len(words)} words, mean confidence {sum(confs) / len(confs):.1f}" if confs
          else "no text found")
    print(f"wrote {txt_path} and {tsv_path}")


if __name__ == "__main__":
    main()
