"""Client for the persistent VL worker (see vl_worker.py).

Two transports exist because no single one covers all three model families:

  http    the mlx-vlm OpenAI-compatible server. Works for PaddleOCR-VL and
          Qwen3.5-VL and returns per-token logprobs, so the vote gets real VL
          confidence. DeepSeek-OCR-2 cannot use it -- generation runs on an
          `asyncio.to_thread` worker whose thread does not own the stream the
          input arrays were created on, giving
          `RuntimeError: There is no Stream(gpu, N) in current thread`.
          Verified in mlx-vlm 0.6.15 and 0.6.17.

  worker  a subprocess holding one model, generating on its main thread. Works
          for DeepSeek-OCR-2. Its `GenerationResult.logprobs` comes back empty,
          so VL confidence is unavailable on this path and is reported as such
          rather than invented.

Crops are handed over as files because `mlx_vlm.generate` takes an image path.
For scanned medical documents those files are patient data, so they are written
to a private per-run directory with restrictive permissions and unlinked
immediately after the call.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2

from hybrid_ocr import VLError, VLReply, downscale_for_vl, record


class WorkerTransport:
    """Speak JSON lines to a vl_worker.py subprocess."""

    def __init__(self, model: str, python_bin: str = ".venv-mlx/bin/python",
                 script: str = "vl_worker.py", revision: str | None = None,
                 startup_timeout: int = 600,
                 default_confidence: float = 0.60):
        self.model = model
        # A declared, deliberately unremarkable prior for backends that cannot
        # report confidence. Not 1.0: an unmeasurable reading must never
        # outweigh one that was actually measured.
        self.default_confidence = default_confidence
        self._lock = threading.Lock()
        self._tmp = tempfile.mkdtemp(prefix="vlcrops-")
        os.chmod(self._tmp, 0o700)
        cmd = [python_bin, script, "--model", model]
        if revision:
            cmd += ["--revision", revision]
        self._err_path = os.environ.get("VL_WORKER_LOG", "vl_worker.err")
        self._err_file = open(self._err_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Keep the worker's diagnostics. Discarding them meant a load
            # failure or a Metal error surfaced only as "VL worker exited during
            # startup", which cost hours of guessing.
            stderr=self._err_file, text=True, bufsize=1)
        deadline = time.time() + startup_timeout
        while True:
            if self.proc.poll() is not None:
                raise VLError(f"VL worker exited during startup (model {model})")
            line = self.proc.stdout.readline()
            if not line:
                if time.time() > deadline:
                    raise VLError(f"VL worker did not become ready within "
                                  f"{startup_timeout}s (model {model})")
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue          # library chatter; the worker keeps it off stdout
            if msg.get("ready"):
                return
            if msg.get("error"):
                raise VLError(f"VL worker startup error: {msg['error']}")
            if time.time() > deadline:
                raise VLError(f"VL worker did not become ready (model {model})")

    def read(self, image_bgr, prompt: str, max_tokens: int = 1024,
             timeout: int = 900) -> VLReply:
        img = downscale_for_vl(image_bgr)
        path = os.path.join(self._tmp, f"crop-{threading.get_ident()}-{time.time_ns()}.png")
        if not cv2.imwrite(path, img):
            raise VLError("could not write the crop for the VL worker")
        try:
            os.chmod(path, 0o600)
            req = json.dumps({"image": path, "prompt": prompt,
                              "max_tokens": max_tokens}) + "\n"
            t0 = time.time()
            # One model, one process: requests are serialised.
            with self._lock:
                if self.proc.poll() is not None:
                    raise VLError("VL worker has exited")
                self.proc.stdin.write(req)
                self.proc.stdin.flush()
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise VLError("VL worker closed its output stream")
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    if "text" in msg or "error" in msg:
                        break
                    if time.time() - t0 > timeout:
                        raise VLError(f"VL worker timed out after {timeout}s")
            record("GPU", "vl:worker", t0, time.time())
            if "error" in msg:
                raise VLError(f"VL worker: {msg['error']}")
            logprobs = msg.get("logprobs") or []
            if logprobs:
                import math
                toks = [("", math.exp(lp)) for lp in logprobs]
                conf = sum(c for _, c in toks) / len(toks)
                available = True
            else:
                toks, conf, available = [], self.default_confidence, False
            return VLReply(msg.get("text") or "", conf, toks,
                           msg.get("finish_reason") == "length",
                           int(msg.get("prompt_tokens") or 0),
                           int(msg.get("completion_tokens") or 0),
                           confidence_available=available)
        finally:
            try:
                os.unlink(path)       # patient data: do not leave it lying around
            except OSError:
                pass

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            shutil.rmtree(self._tmp, ignore_errors=True)
