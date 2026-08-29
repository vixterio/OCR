"""Compare the six OCR methods after each has been run separately.

    .venv/bin/python compare_modes.py output/compare-20260824-181500

There is no ground truth here, so this cannot report accuracy directly. What it
can report is **cost** exactly, and **agreement** as an accuracy proxy: a number
that every method reads the same way is very likely right, and a number only one
method produces is the interesting case. Treat the agreement column as a lead to
investigate, not as a score.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "."
runs = {}
for path in sorted(glob.glob(os.path.join(root, "*", "audit.json"))):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        print(f"  skipping {path}: {exc}")
        continue
    runs[d.get("mode", os.path.basename(os.path.dirname(path)))] = d

if not runs:
    sys.exit(f"no audit.json found under {root}")


def peak_rss(mode: str) -> str:
    # run_matrix.sh writes "<label>.wd"; this looked for ".watchdog", so the
    # peak-RSS column silently printed "?" for every row.
    wd = os.path.join(root, mode.replace("+", "_") + ".wd")
    if not os.path.exists(wd):
        return "?"
    for line in reversed(open(wd, errors="ignore").read().splitlines()):
        if "peak RSS" in line:
            return line.split("peak RSS")[1].split("MB")[0].strip() + " MB"
    return "?"


print(f"\n{'mode':15} {'verify':6} {'pages':5} {'nums':5} {'flag':5} "
      f"{'VL calls':8} {'tokens':8} {'tok/page':8} {'wall':7} {'s/page':7} {'peak RSS':9}")
print("-" * 104)
rows = []
for mode, d in sorted(runs.items()):
    pages = d.get("pages", [])
    npages = max(1, len(pages))
    nums = sum(len(p.get("numbers", [])) for p in pages)
    flag = sum(1 for p in pages for n in p.get("numbers", []) if n.get("needs_review"))
    calls = sum(p.get("vl_calls", 0) for p in pages)
    toks = sum(p.get("vl_prompt_tokens", 0) + p.get("vl_completion_tokens", 0)
               for p in pages)
    wall = d.get("wall_seconds", 0)
    rows.append((mode, d))
    print(f"{mode:15} {str(d.get('verification')):6} {len(pages):5} {nums:5} {flag:5} "
          f"{calls:8} {toks:8} {toks // npages:8} {wall:7.1f} {wall / npages:7.1f} "
          f"{peak_rss(mode):>9}")

# ---- agreement: how many methods read each value the same way ---------------
by_mode = {}
for mode, d in rows:
    vals = Counter()
    for p in d.get("pages", []):
        for n in p.get("numbers", []):
            vals[n.get("display") or n.get("value")] += 1
    by_mode[mode] = vals

everyone = set.intersection(*(set(v) for v in by_mode.values())) if by_mode else set()
union = set().union(*(set(v) for v in by_mode.values())) if by_mode else set()
print(f"\nnumbers read identically by ALL {len(by_mode)} methods: "
      f"{len(everyone)} of {len(union)} distinct values seen anywhere")

support = defaultdict(list)
for mode, vals in by_mode.items():
    for v in vals:
        support[v].append(mode)

singletons = {v: m for v, m in support.items() if len(m) == 1}
if singletons:
    print(f"\nvalues produced by exactly ONE method ({len(singletons)}) — "
          f"either that method alone read it correctly, or it invented it:")
    for v, m in sorted(singletons.items())[:30]:
        print(f"  {v:22} only {m[0]}")

print("\nper-method values missing from the all-method consensus:")
for mode, vals in sorted(by_mode.items()):
    missing = everyone - set(vals)
    extra = {v for v in vals if len(support[v]) == 1}
    print(f"  {mode:15} missing {len(missing):3}  unique {len(extra):3}")

print("\nCaveat: agreement is not truth. All methods can share a failure mode -- a "
      "faint decimal separator lost by every engine looks unanimous. Use this to "
      "pick cases to inspect, then confirm against the scan.")
