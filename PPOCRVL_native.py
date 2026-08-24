"""PaddleOCR-VL using the local ('native') backend — no external server needed.

Memory-instrumented: this machine has 8 GB RAM and Paddle's CPU build has no
bf16/fp16 matmul kernels, so the 0.9B VL model is loaded in float32 (~3.8 GB).
Run this under ./safe_run.sh so a watchdog kills it before the Mac swap-thrashes.
"""
import os
import resource
import sys


def rss_mb():
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def stage(label):
    print(f"[mem] {label:<28} peak RSS = {rss_mb():7.0f} MB", flush=True)


# Paddle exposes no Python thread-count API; OMP_NUM_THREADS / CPU_NUM (set by
# safe_run.sh) are the only knobs, and they must be set before paddle is imported.
stage("startup")

from paddleocr import PaddleOCRVL  # noqa: E402  (import after thread limits)

stage("after import")

pipeline = PaddleOCRVL(
    pipeline_version="v1",
    vl_rec_backend="native",   # run the VL model in-process instead of a vLLM server
    use_queues=False,          # no extra worker queues -> no duplicate model copies
)
stage("after pipeline build")

output = pipeline.predict(
    sys.argv[1] if len(sys.argv) > 1 else "demo.png",
    # Cap vision tokens: fewer pixels -> far less activation memory and much
    # faster on CPU. 1024 patches of 28x28 is plenty for a single page.
    max_pixels=1024 * 28 * 28,
)
stage("after predict")

os.makedirs("output", exist_ok=True)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
stage("after save")
print("DONE")
