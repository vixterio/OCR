"""Launch the mlx-vlm server with a workaround for the DeepSeek-OCR-2 stream bug.

DeepSeek-OCR-2 works correctly under `mlx_vlm.generate` but fails under
`mlx_vlm.server` with:

    RuntimeError: There is no Stream(gpu, 2) in current thread.
    at mlx_vlm/models/deepseekocr_2/deepseekocr_2.py:86
       if mx.sum(global_images).item() == 0:

`.item()` forces evaluation, and MLX streams are per-thread. The server runs
generation via `asyncio.to_thread`, so the model executes on a worker thread that
has no GPU stream of its own. Verified present in mlx-vlm 0.6.15 and 0.6.17.

The fix enters an explicit GPU stream around that method, which is where the
synchronising call lives. It lives here, in a committed file, rather than as an
edit inside `.venv-mlx/` -- the virtualenv is not in version control, so a
site-packages edit would silently disappear on any rebuild.

Remove this file once upstream sets a default stream on its worker threads.

Usage is identical to `python -m mlx_vlm.server`:
    .venv-mlx/bin/python mlx_server_patched.py --model <id> --host 127.0.0.1 --port 8080
"""
import sys

import mlx.core as mx


def apply_deepseek_stream_fix() -> bool:
    try:
        from mlx_vlm.models.deepseekocr_2 import deepseekocr_2 as ds
    except Exception:
        return False
    target = getattr(ds, "Model", None)
    if target is None or not hasattr(target, "get_input_embeddings"):
        return False
    if getattr(target, "_stream_fix_applied", False):
        return True

    original = target.get_input_embeddings

    def get_input_embeddings(self, *args, **kwargs):
        # Binding the GPU stream on *this* thread is what the upstream code
        # assumes but never does.
        with mx.stream(mx.default_stream(mx.gpu)):
            return original(self, *args, **kwargs)

    target.get_input_embeddings = get_input_embeddings
    target._stream_fix_applied = True
    return True


def apply_all() -> list:
    applied = []
    if apply_deepseek_stream_fix():
        applied.append("deepseekocr_2.get_input_embeddings GPU stream")
    return applied


if __name__ == "__main__":
    for name in apply_all():
        print(f"[patch] {name}", file=sys.stderr, flush=True)
    from mlx_vlm.server import main
    main()
