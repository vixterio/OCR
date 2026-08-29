"""Number semantics for clinical documents.

Extracted from hybrid_ocr.py and rewritten, because the previous inline version
corrupted values in ways no amount of cross-engine agreement could detect. Every
rule here exists to fix a specific defect that was reproduced by execution; see
test_numeric.py, which is the regression harness for all of them.

The governing principle is that **inflating a dose is the dangerous direction**,
and that a genuinely ambiguous number must be flagged rather than guessed. A
flagged number costs a reviewer ten seconds; a silently wrong one can reach a
patient.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

# Separators that can never be a decimal point, so they are always grouping.
SPACE_SEPS = "    '’"
CURRENCY = "$€£¥"

# A bare number: optional currency, digits, optional [.,] groups. No sign here --
# signs are handled by the structural pass so that '120-80' cannot become -80.
_NUM_SPACE = r"\d{1,3}(?:[\u00a0\u202f\u2007 '\u2019]\d{3})+(?:[.,]\d+)?"
_NUM_PLAIN = r"\d+(?:[.,]\d+)*"
_NUM_NAKED = r"[.,]\d+"
# Longest-first. Space grouping only counts when it really splits digits into
# threes, so 'in 2024 3 people' stays two numbers; the naked form ('.5') was
# unmatchable by the old regex entirely.
_BARE = rf"(?:{_NUM_SPACE}|{_NUM_PLAIN}|{_NUM_NAKED})"

# Units seen in clinical documents. Order matters: longest first, so 'mcg' wins
# over 'g' and 'mmol/L' over 'L'.
UNITS = [
    "mmol/L", "micromol/L", "umol/L", "µmol/L", "mmol", "mol/L",
    "mg/dL", "mg/dl", "mg/mL", "mg/ml", "mg/kg", "mcg/kg", "ng/mL", "pg/mL",
    "g/dL", "g/dl", "g/L", "IU/L", "U/L", "kU/L", "mEq/L", "mEq",
    "mcg", "µg", "ug", "mg", "kg", "µL", "uL", "mL", "ml", "dL", "L",
    "IU", "iu", "units", "unit", "U", "u", "g", "%",
    "mmHg", "bpm", "kPa", "°C", "C", "cm", "mm", "kcal",
]
_UNIT_RE = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

# Units whose confusion is a 1000x error. Flagged whenever one is present, because
# mcg/mg is a character-level distinction the digit vote cannot see.
RISKY_UNITS = {"mcg", "µg", "ug", "mg", "g", "kg", "IU", "U", "u", "units", "unit"}


@dataclass
class Quantity:
    """One numeric thing found in a line of text."""
    raw: str                       # exactly as it appeared
    kind: str                      # plain | range | ratio | fraction | date | product
    values: list[str]              # canonical surface forms, precision preserved
    unit: str | None = None
    span: tuple[int, int] = (0, 0)
    flags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Comparison key for voting: kind plus numeric values plus unit.

        Numeric comparison ignores trailing-zero differences so '5.0' and '5' vote
        together, but `values` keeps the surface form so the precision difference
        is still visible in the audit.
        """
        return f"{self.kind}:{'/'.join(_canonical(v) for v in self.values)}:{self.unit or ''}"


def _canonical(v: str) -> str:
    """Canonical numeric form for comparison, without scientific notation.

    `Decimal("140").normalize()` is `1.4E+2`, so a sodium of 140 was reaching the
    audit and the clinician's tooltip as "1.4E+2". Integral values are re-quantised
    to a plain integer; fractional values keep normalise's trailing-zero collapse
    so that '5.0' and '5' still compare equal.
    """
    try:
        d = Decimal(v)
    except InvalidOperation:
        return v
    if d == d.to_integral_value():
        return str(d.quantize(Decimal(1)))
    return str(d.normalize())


def _decimal_or_none(s: str):
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def normalise(raw: str) -> tuple[str | None, list[str]]:
    """Canonicalise one bare number. Returns (surface_form, flags).

    Rules, each fixing a reproduced defect:

    * Space and apostrophe separators are always grouping -- they are never a
      decimal point in any locale.
    * When both '.' and ',' appear, the last one is the decimal point. This is
      unambiguous and covers '1,284.50' and '1.284,50' alike.
    * With a single ',' before exactly three digits, treat it as grouping
      ('1,284' -> 1284). Comma-as-decimal with exactly three decimals is rare.
    * With a single '.' before exactly three digits, treat it as **decimal** and
      flag it. This is the fix for the worst defect in the old code, which turned
      '0.125' into '0125'; three-decimal doses are routine in clinical text, so
      the old thousands reading multiplied every one of them by 1000.
    * Trailing zeros are preserved. Precision is clinical information.
    """
    flags: list[str] = []
    s = unicodedata.normalize("NFKC", raw).strip()
    s = re.sub(f"[{re.escape(CURRENCY)}]", "", s)

    grouped_by_space = any(ch in s for ch in SPACE_SEPS)
    for ch in SPACE_SEPS:
        s = s.replace(ch, "")

    if s.startswith((".", ",")):
        # '.5 mg' -- a naked leading decimal, on every error-prone-abbreviation
        # list because it is read as 5. Make it explicit and flag it.
        s = "0" + s.replace(",", ".", 1)
        flags.append("naked_decimal")

    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif has_comma:
        parts = s.split(",")
        if len(parts) > 2:
            s = s.replace(",", "")            # 1,284,567 -- multiple groups
        elif (len(parts) == 2 and len(parts[1]) == 3 and parts[0]
              and parts[0][0] != "0"):
            # Same guard as the dot branch below, and for the same reason: without
            # the leading-zero test '0,125' became '0125', a thousand-fold dose
            # error on a number written the European way. Three decimal places are
            # routine in clinical dosing, and an integer part of '0' can never be
            # a thousands group.
            s = s.replace(",", "")
            flags.append("ambiguous_thousands")
        else:
            s = s.replace(",", ".")           # decimal comma
    elif has_dot:
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")            # 1.284.567 -- multiple groups
        elif len(parts) == 2 and len(parts[1]) == 3 and parts[0] and parts[0][0] != "0":
            # Genuinely ambiguous: '1.284' is 1284 in de/es/it and 1.284 in en.
            # Read it as a decimal, because reading it as thousands is what
            # produced the 1000x dose corruption, and flag it for review.
            flags.append("ambiguous_thousands")

    if grouped_by_space and "." not in s and "," not in s:
        pass                                   # already de-grouped, integer

    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None, flags
    return s, flags


# ---- structural patterns, matched before bare numbers so that a hyphen or -----
# ---- slash between digits is never read as a sign or a separator --------------
_PATTERNS = [
    # ISO and slash/dot dates. Matched first so '2024-05-12' never yields -05.
    ("date", re.compile(r"(?<![\d.,])(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?![\d.,])")),
    ("date", re.compile(r"(?<![\d.,])(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?![\d.,])")),
    # 'BP 120/80', 'Gleason 3+4' style ratios.
    ("ratio", re.compile(rf"(?<![\d.,])({_BARE})\s*/\s*({_BARE})(?![\d.,/])")),
    # '2.5-3.5', '10 - 20' ranges (en dash and hyphen).
    ("range", re.compile(rf"(?<![\d.,])({_BARE})\s*[-–−]\s*({_BARE})(?![\d.,])")),
    # '2 x 500 mg' -- the product is the dose.
    ("product", re.compile(rf"(?<![\d.,])({_BARE})\s*[x×]\s*({_BARE})(?![\d.,])")),
]

_FRACTION_MAP = {"½": "0.5", "¼": "0.25", "¾": "0.75", "⅓": "0.333", "⅔": "0.667"}

# A bare number must not be glued to a preceding letter or digit, so 'HbA1c' does
# not yield 1 and 'B12 450' does not become 12450.
# Two fixed-width lookbehinds. The previous class listed ASCII, Greek and
# Cyrillic but omitted Latin-1 and Latin Extended-A, so the guard that stops
# "B12 450" becoming 12450 did not apply to any accented Latin script:
# "Gęślą987,31" yielded 987,31 and "Kwartał1" yielded 1.
_BARE_RE = re.compile(rf"(?<![0-9.,])(?<![^\W\d_]){_BARE}")
_UNIT_AFTER = re.compile(rf"\s*({_UNIT_RE})(?![A-Za-z])")


def extract(text: str) -> list[Quantity]:
    """Find every quantity in a line, as structure rather than loose digits."""
    if not text:
        return []
    out: list[Quantity] = []
    claimed = [False] * (len(text) + 1)

    def free(a, b):
        return not any(claimed[a:b])

    def claim(a, b):
        for i in range(a, b):
            claimed[i] = True

    for ch, val in _FRACTION_MAP.items():
        for m in re.finditer(re.escape(ch), text):
            if free(*m.span()):
                claim(*m.span())
                out.append(Quantity(m.group(), "fraction", [val], span=m.span(),
                                    flags=["vulgar_fraction"]))

    for kind, rx in _PATTERNS:
        for m in rx.finditer(text):
            if not free(*m.span()):
                continue
            if kind == "date":
                claim(*m.span())
                out.append(Quantity(m.group(), "date", list(m.groups()), span=m.span()))
                continue
            parts, flags = [], []
            for g in m.groups():
                v, f = normalise(g)
                if v is None:
                    parts = None
                    break
                parts.append(v)
                flags += f
            if not parts:
                continue
            claim(*m.span())
            # '1/2 tablet' is a fraction; 'BP 120/80' is a ratio. Distinguish by
            # magnitude: a fraction has a numerator smaller than its denominator.
            if kind == "ratio":
                a, b = _decimal_or_none(parts[0]), _decimal_or_none(parts[1])
                if a is not None and b is not None and a < b and b <= 10:
                    kind = "fraction"
            out.append(Quantity(m.group(), kind, parts, span=m.span(), flags=flags))

    for m in _BARE_RE.finditer(text):
        if not free(*m.span()):
            continue
        v, flags = normalise(m.group())
        if v is None:
            continue
        claim(*m.span())
        unit = None
        um = _UNIT_AFTER.match(text, m.end())
        if um:
            unit = um.group(1)
            if unit in RISKY_UNITS:
                flags = flags + ["risky_unit"]
        out.append(Quantity(m.group(), "plain", [v], unit=unit, span=m.span(), flags=flags))

    out.sort(key=lambda q: q.span[0])
    return out


def keys(text: str) -> list[str]:
    """Comparison keys in reading order, for the vote."""
    return [q.key for q in extract(text)]


# A digit run separated from the next by only a space, where the next run is not a
# 3-digit group, may be a lost decimal separator: '18,7%' misread as '18 7%'.
# Returns the spans involved so the flag can be attached to the numbers concerned
# rather than to every number on the line.
_LOST_SEP = re.compile("(?<=\\d)[" + SPACE_SEPS + "]+(?=(\\d+))")


def lost_separator_spans(text: str) -> list[tuple[int, int]]:
    """Spans where a separator looks lost.

    Two subtleties, both of which produced silent fabrication:

    * The pattern must not CONSUME the digits around the gap. With a consuming
      match, a legitimate thousands group swallowed the digit that anchors the
      next test, so "2 019 75" -- a thousands group followed by a *dropped*
      decimal separator -- was read as 2019 and 75 and flagged as nothing at all.
      That is worse than the case this function was written for, because it is
      silent.
    * The separator class must match the ones normalise() actually treats as
      grouping. It previously knew about three of six, so "1'073,82" Swiss
      grouping was normalised but never checked.
    """
    spans = []
    for m in _LOST_SEP.finditer(text or ""):
        if len(m.group(1)) != 3:
            spans.append((m.start(), m.end() + len(m.group(1))))
    return spans


def format_key(key: str) -> str:
    """Render a comparison key for a human.

    Keys are `kind:values:unit`, which is right for voting and wrong for a
    clinician, so everything user-facing goes through here.
    """
    try:
        kind, values, unit = key.split(":", 2)
    except ValueError:
        return key
    parts = values.split("/")
    if kind == "range" and len(parts) == 2:
        body = f"{parts[0]}–{parts[1]}"
    elif kind == "ratio" and len(parts) == 2:
        body = f"{parts[0]}/{parts[1]}"
    elif kind == "fraction" and len(parts) == 2:
        body = f"{parts[0]}/{parts[1]}"
    elif kind == "product" and len(parts) == 2:
        body = f"{parts[0]}×{parts[1]}"
    elif kind == "date":
        body = "-".join(parts)
    else:
        body = parts[0] if parts else values
    if not unit:
        return body
    # '%' and '°C' are written tight against the number; word units are not.
    return f"{body}{unit}" if unit in ("%", "°C") else f"{body} {unit}"
