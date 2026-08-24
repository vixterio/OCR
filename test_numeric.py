"""Regression harness for numeric.py.

Every case here is a defect that was reproduced by execution in the old inline
implementation, plus the fixture values that must not regress. Run it before any
change to the numeric layer:

    .venv/bin/python test_numeric.py
"""
import sys

from numeric import extract, keys, normalise, lost_separator_spans

FAILURES = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILURES.append((label, got, want))
    print(f"  {'ok  ' if ok else 'FAIL'} {label:52} {got!r}" + ("" if ok else f"  want {want!r}"))


def vals(text):
    """Flat list of canonical values, for terse assertions."""
    return [v for q in extract(text) for v in q.values]


print("\n=== defect 1.0: three-decimal doses must not be multiplied by 1000 ===")
for raw, want in [("0.125", "0.125"), ("0.500", "0.500"), ("2.750", "2.750"),
                  ("1.250", "1.250"), ("12.500", "12.500")]:
    check(f"normalise({raw!r})", normalise(raw)[0], want)
check("Digoxin 0.125 mg -> values", vals("Digoxin 0.125 mg"), ["0.125"])
check("Digoxin 0.125 mg -> unit", extract("Digoxin 0.125 mg")[0].unit, "mg")

print("\n=== defect 1.6: naked leading decimal ===")
check("normalise('.5')", normalise(".5")[0], "0.5")
check("'.5' flagged", "naked_decimal" in normalise(".5")[1], True)
check("'.125 mg' values", vals(".125 mg"), ["0.125"])

print("\n=== defect 1.7: precision is preserved ===")
for raw in ("5.0", "10.00", "0.50"):
    check(f"normalise({raw!r})", normalise(raw)[0], raw)
check("'5.0' and '5' vote together", extract("5.0")[0].key == extract("5")[0].key, True)

print("\n=== defect 1.8: digits bound to a word are not quantities ===")
check("'HbA1c 6.5'", vals("HbA1c 6.5"), ["6.5"])
check("'B12 450'  (was 12450)", vals("B12 450"), ["450"])
check("'CO2 24'", vals("CO2 24"), ["24"])
check("'O2 sat 98%'", vals("O2 sat 98%"), ["98"])

print("\n=== defect 1.9: units are captured and risky ones flagged ===")
a, b = extract("Digoxin 125 mcg")[0], extract("Digoxin 125 mg")[0]
check("125 mcg unit", a.unit, "mcg")
check("125 mg unit", b.unit, "mg")
check("mcg and mg no longer identical", a.key != b.key, True)
check("risky_unit flagged", "risky_unit" in a.flags, True)

print("\n=== defect 1.10 / 1.10b: ranges, ratios, fractions, dates, products ===")
check("'BP 120/80' kinds", [q.kind for q in extract("BP 120/80")], ["ratio"])
check("'BP 120/80' values", vals("BP 120/80"), ["120", "80"])
check("'INR 2.5-3.5' kind", [q.kind for q in extract("INR 2.5-3.5")], ["range"])
check("'INR 2.5-3.5' no negative", vals("INR 2.5-3.5"), ["2.5", "3.5"])
check("'2024-05-12' is one date", [q.kind for q in extract("2024-05-12")], ["date"])
check("'1/2 tablet' is a fraction", [q.kind for q in extract("1/2 tablet")], ["fraction"])
check("'2 x 500 mg' is a product", [q.kind for q in extract("2 x 500 mg")], ["product"])
check("'range 10-20' no negative", vals("range 10-20"), ["10", "20"])

print("\n=== defect 1.10c: lost-separator flag is per-span, not per-line ===")
check("'BP 120 80' spans", lost_separator_spans("BP 120 80"), [(5, 9)])
check("'Итого 2 019,75' no span", lost_separator_spans("Итого 2 019,75"), [])

print("\n=== no regression: English fixture (table_sample.png) ===")
for raw, want in [("1,284.50", "1284.50"), ("10,176.62", "10176.62"),
                  ("18.7", "18.7"), ("6,449.33", "6449.33")]:
    check(f"normalise({raw!r})", normalise(raw)[0], want)

print("\n=== no regression: European fixture (euro_table_sample.png) ===")
for raw, want in [("1.284,50", "1284.50"), ("2 019,75", "2019.75"),
                  ("3 304,25", "3304.25"), ("845,09", "845.09"),
                  ("1 073,82", "1073.82"), ("42,3", "42.3")]:
    check(f"normalise({raw!r})", normalise(raw)[0], want)

print("\n=== ambiguity is flagged, never guessed ===")
v, f = normalise("1.284")
check("normalise('1.284') stays decimal", v, "1.284")
check("normalise('1.284') flagged", "ambiguous_thousands" in f, True)
check("normalise('1,284') is thousands", normalise("1,284")[0], "1284")
check("normalise('1.284.567') is thousands", normalise("1.284.567")[0], "1284567")

print("\n=== reading order is preserved ===")
check("'in 2024 3 people'", vals("in 2024 3 people"), ["2024", "3"])
check("'Wzrost 18,7% — 42,3%'", vals("Wzrost 18,7% — 42,3%"), ["18.7", "42.3"])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for label, got, want in FAILURES:
        print(f"  {label}: got {got!r}, want {want!r}")
    sys.exit(1)
print("all numeric checks passed")
