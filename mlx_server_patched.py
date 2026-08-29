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


def apply_idefics3_pil_processor_fix() -> bool:
    """Let Idefics3 models (Granite Docling) load without torchvision.

    transformers 5.15.1 maps idefics3 to two image-processor backends:

        ("idefics3", {"pil": "Idefics3ImageProcessorPil",
                      "torchvision": "Idefics3ImageProcessor"})

    and resolves both through a lazy module that yields a dummy Placeholder when
    torch is absent. The loader then rejects both as unavailable and raises

        ValueError: Could not load any image processor class ...
        Missing optional dependencies: torchvision

    But the PIL backend genuinely does not need torch: importing
    transformers.models.idefics3.image_processing_pil_idefics3 directly yields a
    real, instantiable class with is_dummy False. Only the lookup is broken, so
    this substitutes the real class when the lazy one comes back a dummy. That
    avoids installing torch and torchvision into the MLX environment purely to
    satisfy a name resolution bug.
    """
    try:
        from transformers.models.auto import image_processing_auto as ipa
    except Exception:
        return False
    if getattr(ipa, "_pil_processor_fix_applied", False):
        return True

    def _real_class(name):
        import importlib
        for module_path in (
            "transformers.models.idefics3.image_processing_pil_idefics3",
            "transformers.models.idefics3.image_processing_idefics3",
        ):
            try:
                mod = importlib.import_module(module_path)
            except Exception:
                continue
            cls = getattr(mod, name, None)
            if isinstance(cls, type) and not getattr(cls, "is_dummy", False):
                return cls
        return None

    original = ipa.get_image_processor_class_from_name

    def get_image_processor_class_from_name(class_name):
        cls = original(class_name)
        if cls is None or getattr(cls, "is_dummy", False):
            real = _real_class(class_name)
            if real is not None:
                return real
        return cls

    ipa.get_image_processor_class_from_name = get_image_processor_class_from_name
    ipa._pil_processor_fix_applied = True
    return True


def apply_all() -> list:
    applied = []
    if apply_deepseek_stream_fix():
        applied.append("deepseekocr_2.get_input_embeddings GPU stream")
    if apply_idefics3_pil_processor_fix():
        applied.append("idefics3 PIL image processor lookup (no torchvision needed)")
    return applied


if __name__ == "__main__":
    for name in apply_all():
        print(f"[patch] {name}", file=sys.stderr, flush=True)
    from mlx_vlm.server import main
    main()
