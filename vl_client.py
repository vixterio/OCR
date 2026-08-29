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

import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2

from hybrid_ocr import (VLError, VLLocalError, VLReply, downscale_for_vl,
                        record)


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
        self._cmd = [python_bin, script, "--model", model]
        if revision:
            self._cmd += ["--revision", revision]
        self._err_path = os.environ.get("VL_WORKER_LOG", "vl_worker.err")
        self._startup_timeout = startup_timeout
        self._restarts = 0
        self._spawn()

    def _spawn(self) -> None:
        """Start the worker subprocess and wait for its ready handshake."""
        self._err_file = open(self._err_path, "a" if self._restarts else "w",
                              buffering=1)
        self.proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Keep the worker's diagnostics. Discarding them meant a load
            # failure or a Metal error surfaced only as "VL worker exited during
            # startup", which cost hours of guessing.
            stderr=self._err_file, text=True, bufsize=1)
        model, startup_timeout = self.model, self._startup_timeout
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

    def _write_crop(self, img, path: str, attempts: int = 3) -> None:
        """Hand one crop to the worker as a file, and say why if that fails.

        cv2.imwrite returns a bare False and discards the reason, which is how a
        page of a patient record was lost with nothing recorded but "could not
        write the crop". Encoding to memory and writing the bytes with Python
        keeps the errno, so ENOSPC (disk full), ENOMEM (no memory to encode) and
        ENOENT (temp directory gone) can be told apart in the audit instead of
        all arriving as False.

        Retried because the cause is usually transient. The observed failure was
        an encode at 2% free memory while 4GB of swap was being written; a second
        attempt a moment later succeeds, and the alternative is dropping the page
        silently. A missing temp directory is repaired rather than retried, since
        waiting will not bring it back.
        """
        last = None
        for attempt in range(attempts):
            try:
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    raise OSError(errno.ENOMEM, "PNG encode failed (out of memory?)")
                os.makedirs(self._tmp, exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(buf.tobytes())
                return
            except (OSError, cv2.error) as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(0.5 * (attempt + 1))
        raise VLLocalError(
            f"could not stage the crop for the VL worker after {attempts} "
            f"attempts: {last}. This is a failure on this machine, not a bad "
            f"reply from the model -- check free memory and disk space.")


    # Errors that mean the worker is unusable but the request is not at fault.
    # A Metal command buffer that overran its deadline says the GPU was busy or
    # the machine was thrashing, not that this page is unreadable -- and MLX
    # leaves the worker's streams wedged afterwards, so the process has to go.
    _FATAL_TO_WORKER = ("has exited", "closed its output stream", "timed out after",
                        "METAL", "Command buffer", "GPU Timeout")

    def read(self, image_bgr, prompt: str, max_tokens: int = 1024,
             timeout: int = 900, max_restarts: int = 2) -> VLReply:
        """One page, surviving a worker that dies underneath it.

        DeepSeek-OCR-2 scored 34% recall on a two-page record because a single
        Metal command-buffer timeout killed the worker on page 1, and page 2 --
        and every page after it, in a longer document -- then failed with "VL
        worker has exited". Nothing restarted it. One transient GPU hiccup cost
        the entire remainder of the document, silently, as a gap in the output.

        Restarting costs a model reload, which is slow, so it is bounded and the
        page is only retried while the failure is one that indicts the worker
        rather than the page. A crop that genuinely cannot be read will fail the
        same way twice, and retrying it forever would be worse than a gap.
        """
        last, started = None, self._restarts
        for attempt in range(max_restarts + 1):
            gen = self._restarts
            try:
                return self._read_once(image_bgr, prompt, max_tokens, timeout)
            except VLLocalError:
                raise                      # staging failure; already retried in place
            except VLError as exc:
                last = exc
                if attempt >= max_restarts:
                    break
                if not any(m in str(exc) for m in self._FATAL_TO_WORKER):
                    break                  # the worker is fine; the reply was not
                sys.stderr.write(
                    f"VL worker died ({exc}); restarting "
                    f"({attempt + 1}/{max_restarts}) and retrying the page\n")
                self._restart(gen)
        # Report restarts actually performed, not the budget. A permanent
        # contract violation restarts nothing, and claiming otherwise sends the
        # reader looking for two model reloads that never happened.
        done = self._restarts - started
        raise VLError(f"{last}" + (f" (after {done} worker restart(s))" if done else ""))

    def _restart(self, gen: int) -> None:
        """Replace the worker, unless another thread already has.

        Without the generation check two callers that failed on the same dead
        worker would each restart: the first spawns a healthy process and the
        second immediately kills it. Only one restart per generation.
        """
        with self._lock:
            if self._restarts != gen:
                return
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=20)
            except Exception:
                pass
            try:
                self._err_file.close()
            except Exception:
                pass
            self._restarts += 1
            self._spawn()

    def _read_once(self, image_bgr, prompt: str, max_tokens: int = 1024,
                   timeout: int = 900) -> VLReply:
        img = downscale_for_vl(image_bgr)
        path = os.path.join(self._tmp, f"crop-{threading.get_ident()}-{time.time_ns()}.png")
        self._write_crop(img, path)
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
