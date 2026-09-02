"""Compare a transcript's HTML against the HTML the document was rendered from.

The generator produced one HTML file per document and then printed it to PDF.
Those PDFs were shuffled into bundles, which is what the pipeline reads. So for
any document we can extract its pages back out (extract_case.py), transcribe
them, and diff the result against the exact markup that produced them -- no hand
labelling anywhere in the loop.

What this does NOT do is score class names. The source corpus has no single
format: three laborbefund documents from three institutions encode the patient
block three mutually incompatible ways -- label/value cells, free-flow <strong>
runs, and a nested definition table -- and use three different table classes
(lab-table, none, data-table). Layout varies per institution by design.
Rewarding a renderer for emitting "patient-info-block" would measure whether we
copied one institution's stylesheet, not whether we read the page.

So it measures three things that are true regardless of house style:

  content    word recall and precision against the source's visible text,
             order-independent, because reading order is a layout choice
  numbers    the quantities, through the same numeric.py the pipeline votes with
  structure  how much of the source's TABULAR shape survived -- tables, rows,
             cells -- since that is what a clinical document carries its data in
             and what a flat stream of paragraphs destroys

    .venv/bin/python compare_html.py work/out/round1 --round work/round1
    .venv/bin/python compare_html.py work/out/round1 --round work/round1 --side-by-side
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from html.parser import HTMLParser

import numeric

# Only elements with a real end tag may open a skip scope. meta and link are
# VOID: handle_endtag never fires for them, so listing them here incremented the
# skip counter forever and silently discarded the whole document -- every text
# score came back 0.0% while the structural counts, which do not consult the
# counter, looked correct. Skipping <head> already covers meta, link and title.
_SKIP = {"script", "style", "head"}
_VOID = {"meta", "link", "br", "hr", "img", "input", "col", "source", "area",
         "base", "embed", "param", "track", "wbr"}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


class _Text(HTMLParser):
    """Visible text plus a census of structural elements."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.tags, self._skip = [], Counter(), 0
        self.rows, self.cells_per_row, self._row_cells = 0, [], 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP and tag not in _VOID:
            self._skip += 1
            return
        self.tags[tag] += 1
        if tag == "tr":
            self._row_cells = 0
        elif tag in ("td", "th"):
            self._row_cells += 1
        if tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "table"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and tag not in _VOID:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "tr":
            self.rows += 1
            self.cells_per_row.append(self._row_cells)

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    @property
    def text(self):
        return re.sub(r"[ \t]+", " ", "".join(self.parts))


def parse(path_or_html: str, is_html: bool = False):
    src = path_or_html if is_html else open(path_or_html, encoding="utf-8").read()
    p = _Text()
    p.feed(src)
    return p


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def words(text: str) -> Counter:
    return Counter(_WORD.findall(norm(text)))


def numbers(text: str) -> Counter:
    out = Counter()
    for k in numeric.keys(text):
        body = k.split(":")[1] if ":" in k else k
        for part in body.split("/"):
            _, part = numeric.split_bound(part)
            v, _f = numeric.normalise(part)
            if v is not None:
                out[v] += 1
    return out


def prf(exp: Counter, got: Counter):
    hit = sum((exp & got).values())
    return (hit / max(1, sum(exp.values())), hit / max(1, sum(got.values())),
            sum(exp.values()) - hit, sum(got.values()) - hit)


# Our own wrapper: the review banner, the mode legend, the filename heading and
# the per-page "Page N" markers. None of it is a transcription, so counting it
# against precision reports the model as inventing text that the renderer added.
# Measured on one document: 78 of 1,334 words, 5.8%, which moved precision by
# about five points. Stripped before comparison and reported, not hidden.
_CHROME = re.compile(
    r'<div class="banner">.*?</div>'
    r'|<p class="legend">.*?</p>'
    r'|<div class="gap">.*?</div>'
    r'|<h1>.*?</h1>'
    r'|<h2>Page \d+</h2>', re.S)


def strip_chrome(html_text: str) -> tuple[str, int]:
    cleaned, n = _CHROME.subn(" ", html_text)
    return cleaned, n


def compare(source_html: str, produced_html: str):
    raw = open(produced_html, encoding="utf-8").read()
    cleaned, n_chrome = strip_chrome(raw)
    s, o = parse(source_html), parse(cleaned, is_html=True)
    sw, ow = words(s.text), words(o.text)
    sn, on = numbers(s.text), numbers(o.text)
    wr, wp, wmiss, wspur = prf(sw, ow)
    nr, np_, nmiss, nspur = prf(sn, on)

    # Tabular shape. A clinical document keeps its data in tables; a transcript
    # that flattens them into paragraphs has lost the row/column association
    # even when every word survives, which is why this is counted separately
    # from the text.
    s_cells = sum(s.tags[t] for t in ("td", "th"))
    o_cells = sum(o.tags[t] for t in ("td", "th"))
    return {
        "word_recall": wr, "word_precision": wp,
        "words_expected": sum(sw.values()), "words_missed": wmiss,
        "num_recall": nr, "num_precision": np_,
        "nums_expected": sum(sn.values()), "nums_missed": nmiss, "nums_spurious": nspur,
        "src_tables": s.tags["table"], "out_tables": o.tags["table"],
        "src_rows": s.rows, "out_rows": o.rows,
        "src_cells": s_cells, "out_cells": o_cells,
        "cell_ratio": (o_cells / s_cells) if s_cells else None,
        "src_headings": sum(s.tags[t] for t in ("h1", "h2", "h3")),
        "out_headings": sum(o.tags[t] for t in ("h1", "h2", "h3")),
        "chrome_stripped": n_chrome,
    }


SIDE_CSS = """
body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f6f7f8;color:#111}
header{padding:14px 18px;background:#fff;border-bottom:1px solid #ccc;position:sticky;top:0}
h1{font-size:1.05rem;margin:0 0 4px}
.stats{font:12px ui-monospace,Menlo,monospace;color:#555}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:0;align-items:start}
.pane{padding:16px 18px;overflow-x:auto;background:#fff;min-height:60vh}
.pane+.pane{border-left:2px solid #ccc;background:#fcfcfd}
.pane h2{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:#666;
         margin:0 0 10px;position:sticky;top:0;background:inherit;padding:4px 0}
.pane table{border-collapse:collapse;font-size:11px;margin:6px 0}
.pane td,.pane th{border:1px solid #ddd;padding:2px 4px}
.pane .num{border-bottom:1px dotted #999}
@media (max-width:900px){.wrap{grid-template-columns:1fr}.pane+.pane{border-left:0;border-top:2px solid #ccc}}
"""


def side_by_side(rec, stats, out_path):
    """One file per document: the source on the left, our transcript on the right."""
    def body_of(path):
        src = open(path, encoding="utf-8").read()
        m = re.search(r"<body[^>]*>(.*)</body>", src, re.S)
        inner = m.group(1) if m else src
        return re.sub(r"<(script|style)\b.*?</\1>", "", inner, flags=re.S | re.I)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>{htmllib.escape(rec['document_id'])} — {htmllib.escape(rec['patient'])}</title>
<style>{SIDE_CSS}</style></head><body>
<header><h1>{htmllib.escape(rec['patient'])} · {htmllib.escape(rec['document_id'])}
 · {htmllib.escape(rec.get('institution') or '')}</h1>
<div class="stats">words {stats['word_recall']*100:.1f}% recall / {stats['word_precision']*100:.1f}% precision
 &nbsp;|&nbsp; numbers {stats['num_recall']*100:.1f}% / {stats['num_precision']*100:.1f}%
 ({stats['nums_missed']} missed, {stats['nums_spurious']} spurious)
 &nbsp;|&nbsp; cells {stats['out_cells']}/{stats['src_cells']}
 &nbsp;|&nbsp; tables {stats['out_tables']}/{stats['src_tables']}</div></header>
<div class="wrap">
<div class="pane"><h2>Source — what the PDF was made from</h2>{body_of(rec['source_html'])}</div>
<div class="pane"><h2>Our transcript</h2>{body_of(stats['_produced'])}</div>
</div></body></html>""")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("outdir", help="e.g. work/out/round1")
    ap.add_argument("--round", required=True, help="e.g. work/round1 (holds index.json)")
    ap.add_argument("--side-by-side", action="store_true",
                    help="also write one openable side-by-side HTML per document")
    args = ap.parse_args()

    # Keyed the way the runner names its output directories. The PDF basename
    # is not unique -- every case has a 00_hausarztbrief -- so keying on it
    # silently matched nothing and reported "nothing to compare".
    index = {f"case{r['case_id']}_{r['document_id']}": r
             for r in json.load(open(f"{args.round}/index.json"))}
    modes = sorted(d for d in os.listdir(args.outdir)
                   if os.path.isdir(os.path.join(args.outdir, d)))
    if not modes:
        sys.exit(f"no mode directories in {args.outdir}")

    sbs_dir = os.path.join(args.outdir, "_side_by_side")
    if args.side_by_side:
        os.makedirs(sbs_dir, exist_ok=True)

    rows, agg = [], {}
    for mode in modes:
        per = []
        for doc, rec in sorted(index.items()):
            produced = os.path.join(args.outdir, mode, doc, "document.html")
            if not os.path.exists(produced) or not rec.get("source_html"):
                continue
            st = compare(rec["source_html"], produced)
            st["_produced"] = produced
            st.update(mode=mode, doc=doc, patient=rec["patient"],
                      document_type=rec["document_type"])
            per.append(st)
            rows.append(st)
            if args.side_by_side:
                side_by_side(rec, st, os.path.join(sbs_dir, f"{mode}__{doc}.html"))
        if per:
            agg[mode] = per

    if not rows:
        sys.exit("nothing to compare -- no document.html found yet")

    print(f"{'mode':9} {'document':30} {'word R':>7} {'word P':>7} "
          f"{'num R':>6} {'num P':>6} {'miss':>5} {'spur':>5} {'cells':>11} {'tables':>7}")
    print("-" * 104)
    for r in rows:
        print(f"{r['mode']:9} {r['doc'][:30]:30} {r['word_recall']*100:6.1f}% "
              f"{r['word_precision']*100:6.1f}% {r['num_recall']*100:5.1f}% "
              f"{r['num_precision']*100:5.1f}% {r['nums_missed']:5} {r['nums_spurious']:5} "
              f"{r['out_cells']:5}/{r['src_cells']:<5} {r['out_tables']:3}/{r['src_tables']:<3}")

    print(f"\n{'mode':9} {'docs':>5} {'word R':>7} {'word P':>7} {'num R':>6} {'num P':>6} "
          f"{'cells kept':>11}")
    print("-" * 60)
    for mode, per in agg.items():
        n = len(per)
        f = lambda k: sum(x[k] for x in per) / n
        cells = sum(x["out_cells"] for x in per) / max(1, sum(x["src_cells"] for x in per))
        print(f"{mode:9} {n:5} {f('word_recall')*100:6.1f}% {f('word_precision')*100:6.1f}% "
              f"{f('num_recall')*100:5.1f}% {f('num_precision')*100:5.1f}% {cells*100:10.0f}%")
    if args.side_by_side:
        print(f"\nside-by-side pages: {sbs_dir}/")


if __name__ == "__main__":
    main()
