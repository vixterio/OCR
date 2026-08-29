"""Regression harness for the fusion vote.

The cases here are the ones that decide whether a correlated engine family can
manufacture a majority, and whether the audit still shows the reading that lost.
Both were defects reproduced by execution, not hypotheticals: the second is the
real 3.412,66 line from fixtures/euro_table_heavy.png, where a lost decimal point
was one part in sixty from winning.

    .venv/bin/python test_vote.py
"""
import sys

from hybrid_ocr import family, family_ballots, vote

FAILURES = []
PRIORS = {"vl": 1.0, "ppocr": 1.0, "tesseract": 1.0}
R, W = "plain:1284.50:", "plain:128450:"      # correct, and the same number with
                                              # its separator lost -- a 100x error


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILURES.append((label, got, want))
    print(f"  {'ok  ' if ok else 'FAIL'} {label:56} {got!r}" + ("" if ok else f"  want {want!r}"))


def winner(per_source, collapse=True):
    return vote(per_source, PRIORS, {R, W}, collapse)["value"]


print("family() maps variants onto one family")
check("padded ppocr variant", family("ppocr:adaptive3x"), "ppocr")
check("unpadded ppocr", family("ppocr"), "ppocr")
check("tesseract variant", family("tesseract:up3x"), "tesseract")

print("\na correlated family must not out-vote the independent engines")
# Four looks at the same pixels through the same recogniser, all wrong together.
# Under per-engine ballots this beat Tesseract and the VL model 4-2.
correlated = {"ppocr": (W, .9), "ppocr:up3x": (W, .9), "ppocr:otsu3x": (W, .9),
              "ppocr:adaptive3x": (W, .9), "tesseract": (R, .85), "vl": (R, .95)}
check("4 wrong ppocr variants lose to tesseract+vl", winner(correlated), R)
check("...and per-engine ballots still lose it", winner(correlated, False), W)

print("\nagreement must not be manufactured, or destroyed")
unanimous = {"ppocr": (R, .9), "ppocr:up3x": (R, .9), "tesseract": (R, .9), "vl": (R, .95)}
check("everyone agrees", winner(unanimous), R)
check("margin is total when unopposed",
      vote(unanimous, PRIORS, {R}, True)["margin_frac"], 1.0)

print("\nthe losing reading stays visible in the audit")
# The real readings from the 3.412,66 line of the heavy euro fixture. Awarding
# the whole family ballot to its internal winner erased '341266' from the
# candidates and drove margin_frac to 1.00, so a number that survived by 1.6%
# looked unanimous. The ballot is split instead.
A, B = "plain:3412.66:", "plain:341266:"
real = {"ppocr": (B, .9406), "ppocr:raw": (A, .9860),
        "ppocr:up3x": (A, .9727), "ppocr:otsu3x": (B, .9558)}
res = vote(real, PRIORS, {A, B}, True)
check("both candidates survive", sorted(res["candidates"]), sorted([A, B]))
check("the near-tie is still visible as a small margin", res["margin_frac"] < 0.05, True)
check("internal split is reported", round(res["family_agreement"]["ppocr"], 2), 0.51)
check("per-engine margin agrees to 4dp", res["margin_frac"],
      vote(real, PRIORS, {A, B}, False)["margin_frac"])

print("\na family's ballot is one ballot, however many variants it has")
one = {"ppocr": (R, .8)}
four = {"ppocr": (R, .8), "ppocr:up3x": (R, .8),
        "ppocr:otsu3x": (R, .8), "ppocr:adaptive3x": (R, .8)}
b1, _ = family_ballots(one)
b4, _ = family_ballots(four)
check("four unanimous variants weigh the same as one",
      round(sum(b4["ppocr"].values()), 6), round(sum(b1["ppocr"].values()), 6))
split = {"ppocr": (R, .8), "ppocr:up3x": (W, .8)}
bs, ags = family_ballots(split)
check("a split family still casts one ballot", round(sum(bs["ppocr"].values()), 6), 0.8)
check("...divided by internal support", round(ags["ppocr"], 3), 0.5)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for label, got, want in FAILURES:
        print(f"  {label}: got {got!r}, want {want!r}")
    sys.exit(1)
print("all vote checks passed")
