"""Run any of the six OCR modes on a PDF, so they can be compared.

    ./start_server.sh deepseek
    ./safe_run.sh run_ocr.py record.pdf --mode deepseek+ocr --outdir output/ds

Modes (see --list-modes for the resolved model on this machine):

    paddle          PaddleOCR-VL alone            block-wise, OTSL tables
    paddle+ocr      + PP-OCRv6 + Tesseract vote
    deepseek        DeepSeek-OCR-2 alone          page-wise, Markdown
    deepseek+ocr    + PP-OCRv6 + Tesseract vote
    qwen            Qwen3.5-VL alone              page-wise, Markdown
    qwen+ocr        + PP-OCRv6 + Tesseract vote

Each mode names one VL model. The server holds one at a time and hot-swaps to
whatever a request asks for, so switching family costs a reload; use
./start_server.sh <family> to preload it. The startup check confirms only that
the model is in the local cache -- /v1/models enumerates the cache, not the
resident model, so it cannot tell you what is loaded.

DeepSeek-OCR-2 does not use the server at all: it runs in a persistent worker
subprocess, because mlx-vlm's server cannot execute it. See vl_client.py.

Note on priority: run GPU-bound modes at normal priority. Metal command buffers
have an execution deadline, and under `nice 10` DeepSeek-OCR-2 fails with
"[METAL] Command buffer execution failed: GPU Timeout" on input it handles fine
at nice 0. safe_run.sh now defaults to NICE=0 for this reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

import hybrid_ocr
import ocr_core
import vl_registry as reg


def server_models(url: str, timeout: int = 5) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("input", nargs="?", help="PDF or image")
    ap.add_argument("--mode", default="paddle+ocr", choices=sorted(reg.MODES))
    ap.add_argument("--list-modes", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--lang", default="eng", help="Tesseract language, e.g. script/Latin")
    ap.add_argument("--layout-model", default="PP-DocLayoutV2")
    ap.add_argument("--render-dpi", type=float, default=300.0)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--ignore-text-layer", action="store_true")
    # hardware
    ap.add_argument("--ram-gb", type=float, default=None,
                    help="override detected RAM; raises model tier, worker counts, "
                         "watchdog ceiling and pixel budget")
    ap.add_argument("--vl-model", default=None, help="override the model id entirely")
    ap.add_argument("--gpu-workers", type=int, default=None)
    ap.add_argument("--vl-max-pixels", type=int, default=None)
    # vote
    ap.add_argument("--w-vl", type=float, default=1.0)
    ap.add_argument("--w-ppocr", type=float, default=1.0)
    ap.add_argument("--w-tesseract", type=float, default=1.0)
    ap.add_argument("--min-families", type=int, default=2)
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--dissent-floor", type=float, default=0.5)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--block-timeout", type=int, default=900)
    ap.add_argument("--allow-gaps", action="store_true")
    ap.add_argument("--vl-line-reads", dest="vl_line_reads", action="store_true",
                    default=None,
                    help="also ask the VL model to re-read every numeric line. "
                         "Defaults ON for block-granularity models (cheap) and OFF "
                         "for page-granularity ones, where the model has already "
                         "read the line as part of the page and the extra calls "
                         "dominated the run")
    ap.add_argument("--no-vl-line-reads", dest="vl_line_reads",
                    action="store_false")
    ap.add_argument("--sequential-lanes", dest="sequential_lanes",
                    action="store_true", default=None,
                    help="finish the VL page call and release the model before "
                         "loading the OCR engines. Defaults ON for page-granularity "
                         "models under 12 GB, where holding both at once exhausts "
                         "memory and shows up as a Metal GPU timeout")
    ap.add_argument("--concurrent-lanes", dest="sequential_lanes",
                    action="store_false")
    ap.add_argument("--transport", choices=("auto", "http", "worker"), default="auto",
                    help="auto uses each family's required transport: http for "
                         "PaddleOCR-VL and Qwen (per-token logprobs available), "
                         "worker for DeepSeek-OCR-2 (the server cannot run it)")
    ap.add_argument("--skip-model-check", action="store_true",
                    help="run even if the server reports a different model")
    args = ap.parse_args()

    if args.list_modes:
        ram = args.ram_gb or reg.detect_ram_gb()
        print(f"detected/declared RAM: {ram:.0f} GB\n")
        for mode in sorted(reg.MODES):
            spec, model, verify, prof = reg.resolve(mode, ram, args.vl_model)
            print(f"  {mode:14} {reg.MODE_HELP[mode]}")
            print(f"  {'':14} model: {model}")
            print(f"  {'':14} {prof.describe()}")
        print("\ntiers per family:")
        for key, spec in reg.SPECS.items():
            print(f"  {key:9} {spec.tier_table()}")
        return

    if not args.input:
        ap.error("input is required unless --list-modes")

    spec, model, verify, prof = reg.resolve(args.mode, args.ram_gb, args.vl_model)
    gpu_workers = args.gpu_workers or prof.gpu_workers
    max_pixels = args.vl_max_pixels or prof.vl_max_pixels
    outdir = args.outdir or f"output/{args.mode.replace('+', '_')}"
    args.variants = prof.variants
    if args.vl_line_reads is None:
        args.vl_line_reads = (spec.granularity == "block")
    if args.sequential_lanes is None:
        args.sequential_lanes = (spec.granularity == "page" and prof.ram_gb < 12)

    print(f"mode {args.mode}  |  {spec.family}  |  {spec.granularity}-granularity"
          f"{'  |  numeric vote ON' if verify else '  |  no verification'}")
    print(f"model {model}")
    print(prof.describe())
    if verify:
        print(f"per-line VL re-reads: {'on' if args.vl_line_reads else 'off'}"
              f"{'' if args.vl_line_reads else ' (page model already read the line)'}")
        if not args.vl_line_reads:
            print(f"  -> numbers are voted by PP-OCRv6 + Tesseract only; the VL model "
                  f"contributes the prose, not the number vote. Pass --vl-line-reads "
                  f"for a three-family vote (much slower on page models).")
        print(f"lanes: {'sequential' if args.sequential_lanes else 'concurrent'}"
              f"{' (VL model released before the OCR engines load)' if args.sequential_lanes else ''}")
    # The watchdog is a separate process, so hand it the ceiling this profile
    # chose instead of printing a number nothing enforces.
    os.environ.setdefault("RSS_LIMIT_MB", str(prof.rss_limit_mb))

    transport_kind = args.transport if args.transport != "auto" else spec.transport
    hybrid_ocr.VL_MAX_PIXELS = max_pixels
    transport = None
    vl_fn = None

    # DeepSeek-OCR-2's Metal command buffers time out at any reduced priority:
    # measured OK at nice 0, and "[METAL] Command buffer execution failed: GPU
    # Timeout" at nice 5 and nice 10 on identical input. It is also the largest
    # model here (2.56 GB against PaddleOCR-VL's 0.7 GB). On a fanless machine
    # those two facts compose badly: the mode can only run by taking the whole
    # machine, which is worth knowing before starting rather than after.
    if spec.key == "deepseek":
        try:
            current_nice = os.nice(0)
        except OSError:
            current_nice = 0
        if current_nice > 0:
            print(f"WARNING: running at nice {current_nice}. DeepSeek-OCR-2 fails with "
                  f"a Metal command-buffer timeout at reduced priority; it needs "
                  f"nice 0. Re-run with NICE=0, or pick another mode on this hardware.")

    if transport_kind == "http":
        loaded = server_models(hybrid_ocr.SERVER_URL)
        if not loaded:
            sys.exit(f"no VL server on {hybrid_ocr.SERVER_URL}. Start it with:\n"
                     f"  ./start_server.sh {spec.key}")
        # /v1/models lists every model in the local cache, not the loaded one, so
        # presence there proves only that the weights exist on disk. The request
        # carries the model name and the server loads it on demand, so a mismatch
        # costs a reload rather than silently using the wrong model.
        if model not in loaded and not args.skip_model_check:
            sys.exit(f"{model} is not in the server's local cache {loaded}.\n"
                     f"Fetch it first with:\n  ./start_server.sh {spec.key}"
                     f"{'' if args.ram_gb is None else f' {args.ram_gb:.0f}'}\n"
                     f"or pass --skip-model-check to try anyway.")
        print(f"transport http ({hybrid_ocr.SERVER_URL})")

        def vl_fn(image, prompt, max_tokens):
            return hybrid_ocr.vl_read(image, "page", max_tokens, args.block_timeout,
                                      True, prompt, model, spec.strip_patterns)
    else:
        import vl_client
        # The worker holds one model and serialises requests behind a lock, so
        # extra pool workers buy nothing and only keep more Metal buffers alive.
        # On 8 GB that matters: DeepSeek-OCR-2 is 2.56 GB against PaddleOCR-VL's
        # 0.7 GB, and over-provisioning here produced Metal command-buffer
        # timeouts rather than an out-of-memory error.
        gpu_workers = 1
        print(f"transport worker (subprocess holding {model}, 1 in flight)")
        transport = vl_client.WorkerTransport(model)

        def vl_fn(image, prompt, max_tokens):
            reply = transport.read(image, prompt, max_tokens, args.block_timeout)
            if spec.strip_patterns and reply.text:
                import re as _re
                for pat in spec.strip_patterns:
                    reply.text = _re.sub(pat, "", reply.text,
                                         flags=_re.MULTILINE | _re.DOTALL).strip()
            return reply

    priors = {"vl": args.w_vl, "ppocr": args.w_ppocr, "tesseract": args.w_tesseract}

    os.umask(0o077)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    from paddleocr import LayoutDetection
    layout = LayoutDetection(model_name=args.layout_model)

    _engines = {}

    def eng_factory():
        # Cached: process_page cannot hand the built engines back (assigning to
        # its parameter does not reach here), so the cache is what stops a
        # rebuild per page.
        if "eng" not in _engines:
            _engines["eng"] = ocr_core.CpuEngines()
        return _engines["eng"]

    # In sequential mode the OCR engines are built after the VL model has been
    # released, so they are never both resident.
    eng = None if (args.sequential_lanes or not verify) else eng_factory()
    pool = ThreadPoolExecutor(max_workers=max(1, gpu_workers), thread_name_prefix="gpu")
    pages = []
    try:
        for idx, img, dpi, source, chars in hybrid_ocr.iter_input_pages(args.input, args):
            pg = ocr_core.process_page(img, idx, eng, pool, spec, model, verify,
                                       args, priors, vl_fn=vl_fn, layout=layout,
                                       eng_factory=eng_factory if verify else None,
                                       after_vl=(transport.close
                                                 if (args.sequential_lanes and transport
                                                     and not args.vl_line_reads)
                                                 else None))
            pg.dpi, pg.source = dpi, source
            pages.append(pg)
            flagged = sum(1 for n in pg.numbers if n.get("needs_review"))
            print(f"  page {idx + 1}: {len(pg.blocks)} blocks, {len(pg.numbers)} numbers, "
                  f"{flagged} need review, {len(pg.incomplete)} region(s) not transcribed",
                  flush=True)
    finally:
        pool.shutdown(wait=True)
        if transport is not None:
            transport.close()

    title = os.path.basename(args.input)
    hybrid_ocr.write_atomic(os.path.join(outdir, "document.html"),
                            ocr_core.render_document(pages, title, spec, model,
                                                     verify, args.mode))
    audit = {
        "input": args.input, "mode": args.mode, "vl_model": model,
        "vl_family": spec.family, "granularity": spec.granularity,
        "verification": verify, "ram_gb": prof.ram_gb,
        # Recorded so a scored run is unambiguous about which configuration
        # produced it -- the whole point of the comparison.
        "vl_line_reads": bool(args.vl_line_reads),
        "sequential_lanes": bool(args.sequential_lanes),
        "variants": list(args.variants),
        "transport": transport_kind,
        "engine_priors": priors if verify else None,
        "wall_seconds": round(time.time() - t0, 2),
        "pages": [{
            "index": p.index, "width": p.width, "height": p.height,
            "dpi": p.dpi, "source": p.source, "vl_calls": p.vl_calls,
            "vl_prompt_tokens": p.vl_prompt_tokens,
            "vl_completion_tokens": p.vl_completion_tokens,
            "page_markdown": p.page_markdown,
            "blocks": [{"index": b.index, "label": b.label, "box": list(b.box),
                        "status": b.status, "note": b.note} for b in p.blocks],
            "numbers": p.numbers,
        } for p in pages],
    }
    hybrid_ocr.write_atomic(os.path.join(outdir, "audit.json"),
                            json.dumps(audit, indent=2, ensure_ascii=False))

    tot = sum(len(p.numbers) for p in pages)
    review = sum(1 for p in pages for n in p.numbers if n.get("needs_review"))
    gaps = sum(len(p.incomplete) for p in pages)
    pt = sum(p.vl_prompt_tokens for p in pages)
    ct = sum(p.vl_completion_tokens for p in pages)
    calls = sum(p.vl_calls for p in pages)
    print(f"\nwrote {outdir}/document.html and audit.json")
    print(f"pages {len(pages)} | numbers {tot} | need review {review} | gaps {gaps}")
    print(f"VL: {calls} calls, {pt} prompt + {ct} completion = {pt + ct} tokens")
    print(f"wall {audit['wall_seconds']}s")
    if not verify:
        print("NOTE: no verification in this mode; every number is a single reading.")
    if gaps and not args.allow_gaps:
        print(f"\nexiting non-zero: {gaps} region(s) not transcribed "
              f"(--allow-gaps to override)")
        sys.exit(2)


if __name__ == "__main__":
    main()
