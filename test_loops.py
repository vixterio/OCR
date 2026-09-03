"""Regression harness for the decode-loop and DocTags defences.

Every case here comes from a reply a model actually produced. The three loop
shapes are different and a rule written for one misses the others, which is why
there are three defences rather than one:

  a tail of content-free lines      Qwen: 1,000 copies of "| | | | | |"
  a run of identical content lines  a repeated real line, mid-document
  a repeated coordinate box         Granite: 33 copies of "Referenzbereich" at
                                    one <loc> tuple, which no token-level
                                    penalty touches

    .venv/bin/python test_loops.py
"""
import sys

from ocr_core import (collapse_degenerate_tail, collapse_repeated_runs,
                      dedupe_doctags, strip_otsl, html_lang)

FAILURES = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILURES.append((label, got, want))
    print(f"  {'ok  ' if ok else 'FAIL'} {label:56} {got!r}" + ("" if ok else f"  want {want!r}"))


print("content-free tail (Qwen's shape)")
# The real reply ended with 999 copies of "| | | | | |" and one "| | | |", so
# anchoring on the last line saw a run of one. The rule is content, not identity.
ragged = "Real line 1\nReal 2\n" + "| | | | | |\n" * 12 + "| | | |"
check("ragged content-free tail is dropped", collapse_degenerate_tail(ragged)[1], 13)
check("the real content survives",
      collapse_degenerate_tail(ragged)[0].splitlines(), ["Real line 1", "Real 2"])
check("a short run is left alone", collapse_degenerate_tail("a\n| |\n| |\n| |")[1], 0)
check("prose is untouched", collapse_degenerate_tail("Hello\nWorld")[1], 0)
# A lab table legitimately repeats a row; it must carry a letter or digit.
check("12 identical REAL rows are kept",
      collapse_degenerate_tail("| Na | 140 | mmol/l |\n" * 12)[1], 0)

print("\nidentical-line run, mid-document")
mid = "Header\n" + "Referenzbereich\n" * 30 + "Footer line\n"
kept, dropped = collapse_repeated_runs(mid)
check("a 30-line identical run is cut", dropped > 0, True)
check("content on both sides survives",
      ("Header" in kept and "Footer line" in kept), True)
check("a legitimately repeated row is kept",
      collapse_repeated_runs("| x | 1 |\n" * 4)[1], 0)

print("\nrepeated coordinate box (Granite's shape)")
LOC_A = "<loc_255><loc_524><loc_315><loc_531>"
LOC_B = "<loc_10><loc_20><loc_30><loc_40>"
dt = ("<doctag>"
      + f"<text>{LOC_B}Real content here</text>"
      + f"<text>{LOC_A}Referenzbereich</text>" * 33
      + "</doctag>")
out, dropped = dedupe_doctags(dt)
check("32 of 33 copies at one box are dropped", dropped, 32)
check("the first copy is kept", out.count("Referenzbereich"), 1)
check("a distinct box is untouched", "Real content here" in out, True)
check("no duplicates means no drops",
      dedupe_doctags(f"<doctag><text>{LOC_A}a</text><text>{LOC_B}b</text></doctag>")[1], 0)
# Two elements cannot share a box, so identical coordinates are the signal. All
# 15 repeated boxes in the observed reply carried byte-identical text, which is
# what makes dropping them safe.
check("distinct text at distinct boxes both survive",
      dedupe_doctags(f"<doctag><text>{LOC_A}one</text><text>{LOC_B}two</text></doctag>")[0]
      .count("</text>"), 2)

print("\nOTSL markers and language")
check("a leaked cell marker is stripped", strip_otsl("| 1.5<ecel> |")[1], 1)
check("a real less-than is preserved", strip_otsl("a < 300 mg")[1], 0)
check("German maps to de", html_lang("deu"), ' lang="de"')
check("an unknown language gets no attribute", html_lang("script/Latin"), "")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for label, got, want in FAILURES:
        print(f"  {label}: got {got!r}, want {want!r}")
    sys.exit(1)
print("all loop/DocTags checks passed")
