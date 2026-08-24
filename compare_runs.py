"""Diff the numbers found by baseline_vl.py and hybrid_ocr.py.

Every difference is either the vote catching a VL error or the vote introducing
one, so this is the file to read when deciding whether the extra machinery pays.

    .venv/bin/python compare_runs.py output/baseline output/hybrid
"""
import json
import sys
from collections import Counter

base_dir, hyb_dir = sys.argv[1], sys.argv[2]
base = json.load(open(f"{base_dir}/numbers.json", encoding="utf-8"))
hyb = json.load(open(f"{hyb_dir}/audit.json", encoding="utf-8"))

b_keys = Counter(n["key"] for n in base["extracted"])
h_keys = Counter(n["value"] for p in hyb["pages"] for n in p["numbers"])

print(f"baseline (stock VL only): {sum(b_keys.values())} numbers, {base['wall_seconds']}s")
print(f"hybrid   (3-engine vote): {sum(h_keys.values())} numbers, {hyb['wall_seconds']}s")

review = [n for p in hyb["pages"] for n in p["numbers"] if n.get("needs_review")]
print(f"hybrid flagged for review: {len(review)}")

only_b = b_keys - h_keys
only_h = h_keys - b_keys
print(f"\nin baseline only ({sum(only_b.values())}):")
for k, c in list(only_b.items())[:25]:
    print(f"  {k}  x{c}")
print(f"\nin hybrid only ({sum(only_h.values())}):")
for k, c in list(only_h.items())[:25]:
    print(f"  {k}  x{c}")

print("\ndisagreements where the vote overrode the VL model:")
n = 0
for p in hyb["pages"]:
    for num in p["numbers"]:
        vl = (num["readings"].get("vl") or {}).get("value")
        if vl and vl != num["value"]:
            n += 1
            print(f"  VL said {vl!r}, vote chose {num['value']!r} "
                  f"(margin {num['margin_frac']:.2f}, "
                  f"{'REVIEW' if num.get('needs_review') else 'accepted'})")
if not n:
    print("  none")
