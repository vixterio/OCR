"""Persistent single-model VL worker, run under .venv-mlx (Python 3.12).

Why this exists instead of only the HTTP server: DeepSeek-OCR-2 cannot run
through `mlx_vlm.server`. Its `get_input_embeddings` calls `.item()`, which
synchronises, and the server executes generation on an `asyncio.to_thread` worker
whose thread does not own the stream the input arrays were created on:

    RuntimeError: There is no Stream(gpu, N) in current thread.

Verified in mlx-vlm 0.6.15 and 0.6.17, and verified NOT to be a model defect --
the same model produces correct output under `mlx_vlm.generate`. Wrapping the
call in a stream context does not help, because the arrays are already bound to
another thread's stream.

This worker loads one model and generates on the main thread, which is the
configuration that works. It also returns everything the vote needs and the HTTP
path could not always supply: per-token logprobs, token counts and finish_reason.

Protocol: one JSON request per line on stdin, one JSON response per line on
stdout. Requests are {"image": path, "prompt": str, "max_tokens": int}.
Responses are {"text", "logprobs", "prompt_tokens", "completion_tokens",
"finish_reason"} or {"error"}.

    .venv-mlx/bin/python vl_worker.py --model mlx-community/DeepSeek-OCR-2-4bit
"""
import argparse
import contextlib
import json
import sys


def _emit(obj):
    """Every protocol message goes to real stdout, one JSON object per line."""
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _note(msg):
    _emit({"note": msg})


def _bundled_processor(model_path):
    """Construct mlx-vlm's own processor for an architecture, or return None.

    `AutoProcessor.from_pretrained` cannot route DeepSeek-OCR-2: the MLX
    repository's `auto_map` names only AutoConfig and AutoModel, so transformers
    has no processor to instantiate and raises

        ValueError: Unrecognized processing class in <snapshot>

    mlx-vlm ships one at models/deepseekocr_2/processing_deepseekocr.py, so this
    finds the `*Processor` class for the architecture and builds it directly.

    `trust_remote_code=False` is explicit, not omitted: omitting it makes
    transformers resolve the repository's dynamic module, which demands torch,
    torchvision, addict and matplotlib, and it also prompts on stdin, which would
    hang a non-interactive worker. With False, no repository code is executed at
    all -- the right answer for a pipeline handling patient data.
    """
    import importlib

    from mlx_vlm.utils import load_config

    try:
        config = load_config(model_path)
        model_type = config.get("model_type")
    except Exception:
        return None
    if not model_type:
        return None
    try:
        module = importlib.import_module(f"mlx_vlm.models.{model_type}")
    except Exception:
        return None
    for name in dir(module):
        if not name.endswith("Processor"):
            continue
        cls = getattr(module, name)
        if hasattr(cls, "from_pretrained"):
            _note(f"using bundled {name}; AutoProcessor cannot route {model_type}")
            return cls.from_pretrained(str(model_path), trust_remote_code=False)
    return None


def install_processor_fallback():
    """Patch only the processor step of mlx-vlm's loader, not the whole loader.

    Assembling the model and processor by hand instead produced a Metal
    command-buffer timeout: `mlx_vlm.load` does more than build the two objects
    (image processor, detokenizer, stopping criteria, and whatever else it grows),
    and reproducing that by hand is guesswork. Substituting just the failing step
    keeps the official path intact.
    """
    import mlx_vlm.utils as U

    if getattr(U, "_processor_fallback_installed", False):
        return
    original = U.load_processor

    def load_processor(model_path, add_detokenizer=True, eos_token_ids=None, **kwargs):
        try:
            return original(model_path, add_detokenizer, eos_token_ids, **kwargs)
        except Exception:
            processor = _bundled_processor(model_path)
            if processor is None:
                raise
            if add_detokenizer:
                detok = U.load_tokenizer(model_path, return_tokenizer=False)
                tok = getattr(processor, "tokenizer", processor)
                processor.detokenizer = detok(tok)
                eos = (eos_token_ids
                       or getattr(tok, "eos_token_ids", None)
                       or getattr(tok, "eos_token_id", None))
                criteria = U.StoppingCriteria(
                    eos, tok,
                    additional_eos_token_ids=getattr(
                        processor, "additional_eos_token_ids", ()))
                if hasattr(processor, "tokenizer"):
                    processor.tokenizer.stopping_criteria = criteria
                else:
                    processor.stopping_criteria = criteria
            return processor

    U.load_processor = load_processor
    U._processor_fallback_installed = True


def load_with_fallback(model_id: str, revision=None):
    import mlx_vlm

    install_processor_fallback()
    kwargs = {"revision": revision} if revision else {}
    return mlx_vlm.load(model_id, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin the model revision hash")
    args = ap.parse_args()

    import mlx_vlm

    # DeepSeek-OCR-2 ships its own processor class, so AutoProcessor cannot build
    # it without executing code from the model repository. That is a real
    # supply-chain consideration for a PHI pipeline: pin the revision and vendor
    # the weights before production use. PaddleOCR-VL and Qwen3.5-VL do not need
    # this, because mlx-vlm implements their architectures natively.
    # mlx-vlm and the bundled processors print progress to stdout ("Add pad
    # token = ...", "Added chat tokens"). That would corrupt the protocol, so
    # stdout is diverted to stderr for the duration of the load.
    with contextlib.redirect_stdout(sys.stderr):
        model, processor = load_with_fallback(args.model, revision=args.revision)

    config = getattr(model, "config", None)
    if config is None:
        config = {}
    _emit({"ready": True, "model": args.model})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _emit({"error": f"bad request json: {exc}"})
            continue
        if req.get("shutdown"):
            break
        try:
          # keep any library chatter off the protocol stream
            prompt = mlx_vlm.apply_chat_template(
                processor, config, req["prompt"], num_images=1)
            with contextlib.redirect_stdout(sys.stderr):
                result = mlx_vlm.generate(
                    model, processor, prompt, image=req["image"],
                    max_tokens=int(req.get("max_tokens", 1024)),
                    temperature=float(req.get("temperature", 0.0)),
                    verbose=False,
                )
            lp = getattr(result, "logprobs", None)
            # logprobs may be per-token arrays; reduce to a scalar per token.
            tokens = []
            if lp is not None:
                try:
                    for entry in lp:
                        v = float(entry) if not hasattr(entry, "__len__") else float(max(entry))
                        tokens.append(v)
                except Exception:
                    tokens = []
            _emit({
                "text": result.text,
                "logprobs": tokens,
                "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(result, "generation_tokens", 0) or 0),
                "finish_reason": getattr(result, "finish_reason", None),
            })
        except Exception as exc:
            _emit({"error": f"{type(exc).__name__}: {exc}"})
        # Deliberately NOT calling mlx_vlm.clear_mlx_streams() here. It releases
        # the streams owned by this thread, which is the thread the model's
        # weights and KV cache are bound to, so the next request fails with
        #     RuntimeError: There is no Stream(gpu, 1) in current thread.
        # Adding it turned deepseek+ocr into a two-engine vote: every per-line
        # VL call errored, the error was swallowed, and the mode reported three
        # families while using two.

    _emit({"bye": True})


if __name__ == "__main__":
    main()
