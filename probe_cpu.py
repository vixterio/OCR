"""Price the CPU engines and their preprocessing variants, without the GPU.

Two questions this answers that a full mode run cannot answer cheaply:

  1. Do the degraded fixtures discriminate? A benchmark where everything scores
     100% ranks nothing. This walks the severity ladder and shows where each
     engine actually starts to fail.
  2. Are the four PP-OCR preprocessing variants worth their cost, and do they
     fail independently? hybrid_ocr runs all four on every line and lets them
     vote. If they always agree, three of them are pure cost; if they disagree,
     the vote must weight them as one family rather than four witnesses.

No VL model is loaded, so this runs in seconds on any machine and cannot put the
laptop into swap. Every number it reports is per-engine recall against the same
ground truth evaluate.py uses.

    .venv/bin/python probe_cpu.py euro_table_sample.png fixtures/euro_table_*.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import cv2

import evaluate
import hybrid_ocr
import numeric


def canon_set(text: str) -> Counter:
    """The canonical numbers in a string, as a multiset."""
    out = Counter()
    for k in numeric.keys(text):
        body = k.split(":")[1] if ":" in k else k
        for part in body.split("/"):
            c = evaluate.canon(part)
            if c is not None:
                out[c] += 1
    return out


def probe(path: str, eng, lang: str):
    truth_pages = evaluate.truth_for(os.path.basename(path))
    if truth_pages is None:
        print(f"  {path}: no ground truth")
        return None
    expected = Counter(c for c in (evaluate.canon(t) for t in truth_pages[0]) if c)

    img = cv2.imread(path)
    if img is None:
        print(f"  {path}: unreadable")
        return None

    t0 = time.time()
    polys = eng.lines(img)
    det_s = time.time() - t0

    # Per engine variant: everything it read, and what it cost.
    found = defaultdict(Counter)
    cost = defaultdict(float)
    # Per line: what each variant said, to measure whether they agree.
    per_line = []

    for poly in polys:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x1, x2 = max(0, min(xs)), min(img.shape[1], max(xs))
        y1, y2 = max(0, min(ys)), min(img.shape[0], max(ys))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = img[y1:y2, x1:x2]
        says = {}
        for name, variant in hybrid_ocr.preprocess_variants(crop):
            t = time.time()
            text, _ = eng.rec_line(variant)
            cost[f"ppocr:{name}"] += time.time() - t
            found[f"ppocr:{name}"] += canon_set(text)
            says[f"ppocr:{name}"] = text
            if name == "raw":
                t = time.time()
                ttext, _ = eng.tess_line(variant, lang)
                cost["tesseract:raw"] += time.time() - t
                found["tesseract:raw"] += canon_set(ttext)
                says["tesseract:raw"] = ttext
        per_line.append(says)

    return {"path": path, "expected": expected, "found": found, "cost": cost,
            "det_s": det_s, "lines": len(polys), "per_line": per_line}


def short(engine: str) -> str:
    """'ppocr:adaptive3x' -> 'pp/adapt'. Keeps the engine, which is the half that
    matters: 'ppocr:raw' and 'tesseract:raw' are different engines, and printing
    both as 'raw' made two columns indistinguishable."""
    fam, _, var = engine.partition(":")
    return f"{fam[:2]}/{var[:7]}"


def agreement(per_line):
    """How often the four PP-OCR variants produce the same digits on a line.

    The vote treats each variant as a witness. Witnesses who always say the same
    thing carry no extra information -- and if they are wrong together, a vote
    that counts them separately will out-vote a correct reading four to one.
    """
    names = [f"ppocr:{n}" for n in ("raw", "up3x", "otsu3x", "adaptive3x")]
    same = uniq = 0
    seen = 0
    for says in per_line:
        # Compare canonical VALUES, not digit signatures. digit_signature strips
        # separators by design, so it scores "1.284,50" and "1,284.50" as the
        # same read -- and separator confusion is precisely the failure that took
        # recall from 100% to 56% down this ladder. Measured on signatures, the
        # variants looked 95% correlated; the question the vote actually asks is
        # whether they agree on the number, so that is what this counts.
        reads = {n: frozenset(canon_set(says[n]).items()) for n in names if n in says}
        if not reads:
            continue
        seen += 1
        distinct = set(reads.values())
        same += (len(distinct) == 1)
        uniq += len(distinct)
    return same, seen, uniq / max(1, seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--lang", default="script/Latin")
    args = ap.parse_args()

    eng = hybrid_ocr.CpuEngines()
    results = []
    for path in args.images:
        r = probe(path, eng, args.lang)
        if r:
            results.append(r)
    if not results:
        sys.exit("nothing probed")

    engines = sorted({e for r in results for e in r["found"]})
    print(f"\n{'fixture':26} {'lines':>5} " + " ".join(f"{short(e):>10}" for e in engines))
    print("-" * (33 + 11 * len(engines)))
    for r in results:
        exp = sum(r["expected"].values())
        cells = []
        for e in engines:
            hit = sum((r["expected"] & r["found"][e]).values())
            cells.append(f"{100.0*hit/exp:9.0f}%")
        print(f"{os.path.basename(r['path']):26} {r['lines']:5} " + " ".join(cells))

    print(f"\n{'fixture':26} {'variants agree':>15} {'distinct reads/line':>21}")
    print("-" * 64)
    for r in results:
        same, total, uniq = agreement(r["per_line"])
        print(f"{os.path.basename(r['path']):26} {same:8}/{total:<6} {uniq:20.2f}")

    print(f"\n{'engine':16} {'seconds':>9}  {'share':>6}")
    print("-" * 35)
    tot = sum(sum(r["cost"].values()) for r in results)
    agg = Counter()
    for r in results:
        agg.update(r["cost"])
    for e, s in agg.most_common():
        print(f"{e:16} {s:8.2f}s {100.0*s/max(tot,1e-9):5.1f}%")
    print(f"{'TOTAL':16} {tot:8.2f}s")


if __name__ == "__main__":
    main()
