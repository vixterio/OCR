"""Vision-language model registry: which model, which prompts, which hardware.

Three model families are supported, each with size tiers. The default tier is
chosen from available RAM, so an 8 GB laptop and a 64 GB server run the same
command and get the largest model that actually fits.

Two facts drive the whole design and were verified against HuggingFace configs:

  * PaddleOCR-VL is a *block* model. It expects a layout pass to hand it one
    region at a time with a terse task prompt, and it emits OTSL for tables.
  * DeepSeek-OCR-2 and Qwen3.5-VL are *page* models. They take a whole page and
    emit Markdown directly. Feeding them one block at a time wastes their layout
    ability and multiplies the per-request overhead.

That difference is why `granularity` exists rather than one prompt table.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class VLSpec:
    key: str
    family: str
    granularity: str          # 'block' (needs layout) | 'page' (self-directing)
    output: str               # 'otsl' | 'markdown'
    tiers: dict               # min RAM GB -> model id
    page_prompt: str = ""
    line_prompt: str = ""
    block_prompts: dict = field(default_factory=dict)
    max_tokens_page: int = 4096
    max_tokens_line: int = 256
    max_pixels: int = 1024 * 28 * 28
    strip_patterns: tuple = ()
    # 'http' = mlx-vlm OpenAI server (gives per-token logprobs, so the vote gets
    # real VL confidence). 'worker' = persistent subprocess, required for models
    # the server cannot run. See vl_client.py for why DeepSeek needs 'worker'.
    transport: str = "http"
    notes: str = ""

    def model_for(self, ram_gb: float) -> str:
        """Largest tier whose RAM floor the machine meets."""
        eligible = [(need, mid) for need, mid in sorted(self.tiers.items()) if ram_gb >= need]
        if not eligible:
            return self.tiers[min(self.tiers)]
        return eligible[-1][1]

    def tier_table(self) -> str:
        return ", ".join(f"{need}GB+: {mid.split('/')[-1]}"
                         for need, mid in sorted(self.tiers.items()))


PADDLE = VLSpec(
    key="paddle",
    family="PaddleOCR-VL 0.9B",
    granularity="block",
    output="otsl",
    tiers={
        0: "mlx-community/PaddleOCR-VL-4bit",
        12: "mlx-community/PaddleOCR-VL-8bit",
        24: "mlx-community/PaddleOCR-VL-bfloat16",
    },
    # Mirrors paddlex/inference/pipelines/paddleocr_vl/pipeline.py:308-330.
    block_prompts={"table": "Table Recognition:", "chart": "Chart Recognition:",
                   "formula": "Formula Recognition:", "default": "OCR:"},
    line_prompt="OCR:",
    max_tokens_page=4096,
    notes=("Purpose-built for document blocks and very small (0.9B), so it is the "
           "cheapest per page. Trained predominantly on Chinese and English; weak "
           "on Greek and Cyrillic prose. Emits OTSL table markup, which needs the "
           "paddlex converter to become HTML. Measured 4-bit vs 8-bit: no "
           "difference, so precision is not its bottleneck."),
)

DEEPSEEK = VLSpec(
    key="deepseek",
    family="DeepSeek-OCR-2",
    granularity="page",
    output="markdown",
    tiers={
        0: "mlx-community/DeepSeek-OCR-2-4bit",
        12: "mlx-community/DeepSeek-OCR-2-6bit",
        16: "mlx-community/DeepSeek-OCR-2-8bit",
        32: "mlx-community/DeepSeek-OCR-2-bf16",
    },
    page_prompt="Convert the document to markdown.",
    line_prompt="Free OCR.",
    max_tokens_page=8192,
    max_tokens_line=256,
    # DeepSeek's optical-compression encoder is built for dense full pages, so it
    # gets a larger budget than the block models.
    max_pixels=1280 * 28 * 28,
    transport="worker",
    # Grounding mode emits <|ref|>/<|det|> spans; strip them if they appear.
    strip_patterns=(r"<\|ref\|>.*?<\|/ref\|>", r"<\|det\|>.*?<\|/det\|>",
                    r"<\|grounding\|>"),
    notes=("Designed for optical context compression: it encodes a whole page into "
           "far fewer vision tokens than a general VLM, which is why it is the "
           "cheapest per *page* despite being larger than PaddleOCR-VL. Emits "
           "Markdown directly, so no OTSL conversion is needed. Purpose-trained on "
           "documents, including dense multi-column layouts."),
)

QWEN = VLSpec(
    key="qwen",
    family="Qwen3.5-VL (Qwen3.8 on larger hardware)",
    granularity="page",
    output="markdown",
    tiers={
        0: "mlx-community/Qwen3.5-2B-MLX-4bit",
        10: "mlx-community/Qwen3.5-4B-MLX-4bit",
        20: "mlx-community/Qwen3.5-9B-MLX-4bit",
        48: "Qwen/Qwen3.8-27B",
    },
    page_prompt=("Transcribe this document page into Markdown. Preserve the reading "
                 "order and represent every table as a Markdown table. Transcribe "
                 "numbers exactly as printed, including decimal separators. Do not "
                 "summarise, explain, or add commentary. Output only the "
                 "transcription."),
    line_prompt=("Transcribe the text in this image exactly as printed. Output only "
                 "the text, with no commentary."),
    max_tokens_page=8192,
    max_tokens_line=128,
    # A general VLM may prepend chatter despite the instruction.
    strip_patterns=(r"^\s*(?:Here(?:'s| is)[^\n:]*:|```(?:markdown)?|```)\s*$",),
    notes=("A general-purpose instruction-following VLM rather than an OCR model. "
           "That is its strength and its weakness: it handles unusual layouts and "
           "follows instructions about output format, but it can paraphrase, "
           "summarise or refuse, and it is the most likely of the three to 'tidy' a "
           "number. Broadest language coverage of the three. Qwen3.8-27B is the "
           "newest and needs roughly 48 GB; the 9B MLX build is text-only, so the "
           "usable ladder is 2B -> 4B -> 9B(Qwen3.5) -> 27B(Qwen3.8)."),
)

SPECS = {s.key: s for s in (PADDLE, DEEPSEEK, QWEN)}

MODES = {
    "paddle":          ("paddle", False),
    "paddle+ocr":      ("paddle", True),
    "deepseek":        ("deepseek", False),
    "deepseek+ocr":    ("deepseek", True),
    "qwen":            ("qwen", False),
    "qwen+ocr":        ("qwen", True),
}

MODE_HELP = {
    "paddle": "PaddleOCR-VL alone (block-wise, OTSL tables)",
    "paddle+ocr": "PaddleOCR-VL + PP-OCRv6 + Tesseract, numbers resolved by vote",
    "deepseek": "DeepSeek-OCR-2 alone (page-wise, Markdown)",
    "deepseek+ocr": "DeepSeek-OCR-2 + PP-OCRv6 + Tesseract, numbers resolved by vote",
    "qwen": "Qwen3.5-VL alone (page-wise, Markdown)",
    "qwen+ocr": "Qwen3.5-VL + PP-OCRv6 + Tesseract, numbers resolved by vote",
}


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #
def detect_ram_gb() -> float:
    """Total physical RAM in GB, on macOS or Linux."""
    env = os.environ.get("OCR_RAM_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


@dataclass
class Profile:
    """Resource budget derived from RAM, so one command scales across machines."""
    ram_gb: float
    gpu_workers: int
    cpu_workers: int
    rss_limit_mb: int
    vl_max_pixels: int
    variants: tuple

    @classmethod
    def for_ram(cls, ram_gb: float, spec: VLSpec | None = None):
        # Deliberately conservative. Measured on 8 GB: one page peaks ~1.3 GB for
        # the pipeline plus the VL server, and the binding constraint is memory,
        # not cores. VL calls also serialise on the device (measured 1.01-1.04x
        # from 2-6 workers), so extra GPU workers buy cross-lane overlap only.
        if ram_gb < 12:
            gpu, cpu, rss = 2, 1, 3800
            variants = ("raw", "up3x", "otsu3x")
        elif ram_gb < 24:
            gpu, cpu, rss = 3, 2, 9000
            variants = ("raw", "up3x", "otsu3x", "adaptive3x")
        elif ram_gb < 48:
            gpu, cpu, rss = 4, 4, 20000
            variants = ("raw", "up3x", "otsu3x", "adaptive3x")
        else:
            gpu, cpu, rss = 6, 8, 48000
            variants = ("raw", "up3x", "otsu3x", "adaptive3x")
        px = spec.max_pixels if spec else 1024 * 28 * 28
        if ram_gb >= 24:
            px = int(px * 1.5)      # more headroom for the encoder
        return cls(ram_gb, gpu, cpu, rss, px, variants)

    def describe(self) -> str:
        # cpu_workers is reported as a target, not a description: the CPU lane is
        # still a single thread because Paddle predictors are not thread-safe, so
        # using it needs worker processes rather than a bigger number here.
        return (f"{self.ram_gb:.0f} GB RAM -> {self.gpu_workers} VL worker(s), "
                f"watchdog {self.rss_limit_mb} MB, "
                f"{len(self.variants)} pre-processing variants "
                f"(CPU lane is 1 thread regardless; see notes)")


def resolve(mode: str, ram_gb: float | None = None, model_override: str | None = None):
    """(spec, model_id, verify_numbers, profile) for a mode."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    key, verify = MODES[mode]
    spec = SPECS[key]
    ram = ram_gb if ram_gb else detect_ram_gb()
    model = model_override or spec.model_for(ram)
    return spec, model, verify, Profile.for_ram(ram, spec)
