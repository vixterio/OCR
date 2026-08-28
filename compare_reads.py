"""Answer one question from the matrix: does a mode do better with per-line VL
reads on or off?

Pairs each +ocr run's on/off variants and reports the delta in the terms that
matter -- missed and spurious numbers first, cost second, because a fabricated or
lost number in a clinical document is not tradeable against seconds.

    .venv/bin/python compare_reads.py output/matrix
"""
import glob
import json
import os
import sys

from evaluate import score

root = sys.argv[1] if len(sys.argv) > 1 else "output/matrix"
runs = {}
for p in sorted(glob.glob(os.path.join(root, "*", "audit.json"))):
    r = score(p)
    if r:
        runs[os.path.basename(os.path.dirname(p))] = r

families = ["paddle", "deepseek", "qwen", "granite"]
print(f"{'family':10} {'reads':5} {'recall':>7} {'prec':>7} {'missed':>7} {'spurious':>9} "
      f"{'review':>7} {'calls':>6} {'tokens':>7} {'wall':>7}")
print("-" * 84)
verdicts = []
for fam in families:
    on, off = runs.get(f"{fam}_ocr_on"), runs.get(f"{fam}_ocr_off")
    for label, r in (("on", on), ("off", off)):
        if not r:
            print(f"{fam:10} {label:5}   (missing)")
            continue
        t = r["totals"]
        print(f"{fam:10} {label:5} {r['recall']*100:6.1f}% {r['precision']*100:6.1f}% "
              f"{t['missed']:7} {t['spurious']:9} {t['review']:7} "
              f"{r['vl_calls']:6} {r['tokens']:7} {r['wall'] or 0:6.1f}s")
    if on and off:
        d_missed = on["totals"]["missed"] - off["totals"]["missed"]
        d_spur = on["totals"]["spurious"] - off["totals"]["spurious"]
        d_rev = on["totals"]["review"] - off["totals"]["review"]
        d_tok = on["tokens"] - off["tokens"]
        d_wall = (on["wall"] or 0) - (off["wall"] or 0)
        # Accuracy first: reads-on only earns its cost if it loses or fabricates
        # fewer numbers. Equal accuracy means the cheaper option wins.
        if d_missed < 0 or d_spur < 0:
            verdict = "ON  (fewer missed/spurious)"
        elif d_missed > 0 or d_spur > 0:
            verdict = "OFF (on is worse on accuracy)"
        else:
            verdict = f"OFF (identical accuracy, saves {d_tok} tokens / {d_wall:.0f}s)"
        verdicts.append((fam, verdict, d_missed, d_spur, d_rev, d_tok, d_wall))
        print(f"{'':10} delta  missed {d_missed:+d}  spurious {d_spur:+d}  "
              f"review {d_rev:+d}  tokens {d_tok:+d}  wall {d_wall:+.1f}s   -> {verdict}")
    print()

print("=" * 84)
print("RECOMMENDED DEFAULT PER FAMILY")
for fam, verdict, dm, ds, dr, dt, dw in verdicts:
    print(f"  {fam:10} {verdict}")
if not verdicts:
    print("  (no paired runs found)")
