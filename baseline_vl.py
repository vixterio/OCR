"""Baseline: PaddleOCR-VL exactly as PaddlePaddle intends, output as HTML.

This is the comparison arm for hybrid_ocr.py. It runs the packaged
`PaddleOCRVL` pipeline with no voting, no pre-processing variants and no numeric
verification, and writes the pipeline's own HTML via its native `save_to_html`.

Use it to answer "is the extra machinery earning its cost?" -- run both on the
same document and diff the numbers. The point of comparison is not prose quality
(the VL model is good at prose); it is whether any number differs, because every
difference is either the vote catching a VL error or the vote introducing one.

The pipeline accepts a PDF directly, so no page loop is needed here.

Usage:
    ./start_server.sh
    ./safe_run.sh baseline_vl.py record.pdf --outdir output/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import _compat

BASELINE_CSS = """
:root { --bg:#fff; --fg:#111; --line:#bbb; --head:#f2f2f2; --note:#555; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8e8e8; --line:#444; --head:#22252b; --note:#aaa; }
}
body { font:16px/1.65 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;
       max-width:64rem; margin:2rem auto; padding:0 1rem;
       background:var(--bg); color:var(--fg); }
table { border-collapse:collapse; margin:1rem 0; width:100%; }
th,td { border:1px solid var(--line); padding:.35rem .6rem; text-align:right; }
th { background:var(--head); }
td:first-child, th:first-child { text-align:left; }
.page { border-top:3px solid var(--line); margin-top:2.5rem; padding-top:.5rem; }
.banner { border:2px solid var(--line); background:var(--head); color:var(--note);
          padding:.8rem 1rem; margin:1rem 0; }
"""


def html_escape(s: str) -> str:
    from html import escape
    return escape(s or "")


def render_page(index: int, blocks: list) -> str:
    """Render one page of stock-pipeline output, verbatim."""
    try:
        from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html
    except Exception:
        convert_otsl_to_html = None
    parts = []
    for b in blocks:
        # parsing_res_list holds PaddleOCRVLBlock objects (result.py:65-93) with
        # .label/.content attributes, not dicts.
        content = (getattr(b, "content", None) or "").strip()
        label = getattr(b, "label", None) or "text"
        if not content:
            continue
        if "<fcel>" in content or "<ecel>" in content:
            table = convert_otsl_to_html(content) if convert_otsl_to_html else None
            parts.append(table if table else f"<pre>{html_escape(content)}</pre>")
        elif label in ("doc_title",):
            parts.append(f"<h2>{html_escape(content)}</h2>")
        elif label in ("paragraph_title",):
            parts.append(f"<h3>{html_escape(content)}</h3>")
        else:
            parts.append(f"<p>{html_escape(content)}</p>")
    return (f'<section class="page" id="page-{index + 1}">'
            f"<h2>Page {index + 1}</h2>" + "\n".join(parts) + "</section>")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="PDF or image")
    ap.add_argument("--outdir", default="output/baseline")
    ap.add_argument("--pipeline-version", default="v1", choices=["v1", "v1.5", "v1.6"])
    ap.add_argument("--backend", default="mlx-vlm-server",
                    help="mlx-vlm-server (Mac dev), vllm-server (Linux prod), "
                         "llama-cpp-server, sglang-server, fastdeploy-server, native")
    ap.add_argument("--server-url", default=os.environ.get(
        "VL_SERVER_URL", "http://127.0.0.1:8080/v1"))
    ap.add_argument("--model-name", default=os.environ.get(
        "VL_MODEL", "mlx-community/PaddleOCR-VL-4bit"))
    args = ap.parse_args()

    if _compat.apply():
        print("[compat] applied Python 3.9 asyncio semaphore fix", flush=True)

    from paddleocr import PaddleOCRVL

    os.umask(0o077)
    os.makedirs(args.outdir, exist_ok=True)

    kwargs = dict(pipeline_version=args.pipeline_version,
                  vl_rec_backend=args.backend,
                  vl_rec_max_concurrency=1,
                  use_queues=False)
    if args.backend != "native":
        kwargs.update(vl_rec_server_url=args.server_url,
                      vl_rec_api_model_name=args.model_name)

    t0 = time.time()
    pipeline = PaddleOCRVL(**kwargs)
    results = pipeline.predict(args.input)
    elapsed = time.time() - t0

    numbers, pages, page_html = [], 0, []
    import numeric
    for i, res in enumerate(results):
        pages += 1
        stem = f"page_{i + 1}"
        # paddlex's own save_to_html is table-only: PaddleOCRVLResult._to_html
        # (result.py:220-234) returns HTML for `table_res_list` alone, which this
        # pipeline never populates, so it raises IndexError on a normal page. The
        # pipeline's real full-page output is markdown, so that is what gets
        # rendered here -- faithfully, with no voting and no annotation.
        res.save_to_markdown(save_path=os.path.join(args.outdir, f"{stem}.md"))
        res.save_to_json(save_path=os.path.join(args.outdir, f"{stem}.json"))
        page_html.append(render_page(i, res.get("parsing_res_list") or []))
        # Extract the numbers the stock pipeline produced, using the same parser
        # the hybrid uses, so the two are directly comparable.
        for blk in (res.get("parsing_res_list") or []):
            for q in numeric.extract(getattr(blk, "content", None) or ""):
                numbers.append({"page": i, "block_label": getattr(blk, "label", None),
                                "kind": q.kind, "values": q.values,
                                "unit": q.unit, "key": q.key, "raw": q.raw})

    doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>{html_escape(os.path.basename(args.input))} (baseline)</title>"
           f"<style>{BASELINE_CSS}</style></head><body>"
           f"<h1>{html_escape(os.path.basename(args.input))}</h1>"
           "<div class=\"banner\">Stock PaddleOCR-VL output. Every number here is a "
           "single VL reading: no cross-engine vote, no verification, no confidence "
           "shown. Provided for comparison only.</div>"
           + "".join(page_html) + "</body></html>")
    with open(os.path.join(args.outdir, "document.html"), "w", encoding="utf-8") as fh:
        fh.write(doc)

    summary = {"input": args.input, "pipeline_version": args.pipeline_version,
               "backend": args.backend, "model": args.model_name,
               "pages": pages, "numbers": len(numbers),
               "wall_seconds": round(elapsed, 2), "extracted": numbers}
    path = os.path.join(args.outdir, "numbers.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\nbaseline: stock PaddleOCR-VL {args.pipeline_version} via {args.backend}")
    print(f"pages {pages} | numbers {len(numbers)} | wall {summary['wall_seconds']}s")
    print(f"wrote {args.outdir}/document.html (rendered from the pipeline's markdown)")
    print(f"wrote {args.outdir}/page_N.md and page_N.json (the pipeline's own output)")
    print(f"wrote {path}")
    print("\nNo voting, no verification: every number here is one VL reading.")


if __name__ == "__main__":
    main()
