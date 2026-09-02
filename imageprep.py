"""Measure a page, then correct only what is measurably wrong with it.

Every correction in this file was proposed, costed and then argued against
before being written, because document preprocessing is the part of an OCR
pipeline most likely to make things quietly worse. Two measured results shaped
it:

  Binarisation is not a stroke-erosion problem, it is a precision collapse.
  On a heavily degraded fixture, stroke-pixel recall stays at 99.1-99.6% for
  Otsu, Sauvola and adaptive thresholding -- nothing is being erased. What
  collapses is precision: 40.8% (Otsu), 45.8% (Sauvola), 26.8% (adaptive), with
  connected-component counts going from 215 to 1799, 1880 and 8808. The
  thresholds promote speckle into ink. That is why otsu3x scored 55% and
  adaptive3x 64% where raw and up3x scored 100%, and it is why nothing here
  emits a 1-bit image. Sauvola, Niblack, Wolf, NICK and the deep binarisers are
  not substitutes; they fail the same way.

  Expensive denoising does not earn its cost. fastNlMeansDenoising at h=7 costs
  1.76 s/page on an 8.7 MP page -- 17% of Granite's entire 10.2 s/page budget --
  and leaves the blob explosion in place, because it preserves speckle as
  structure by design. A 3x3 median costs 1.6 ms and does more for this failure.

So the design is: a cheap probe that measures the page, a policy that turns
measurements into a plan, and corrections that are grayscale-only and bounded.
Anything that cannot be justified by a measurement does not run.

  ALWAYS      probe, skew estimation, content-box crop
              -- measurement is free of side effects, and the crop only ever
                 spends the model's fixed pixel budget on ink instead of margin
  CONDITIONAL deskew rotation, background division, CLAHE, median denoise
              -- each resamples or rewrites pixels, so each needs evidence
  NEVER       any binarisation, fastNlMeansDenoising, whole-page deconvolution

    from imageprep import probe_page, plan_for, apply_plan
    p = probe_page(img)
    plan = plan_for(p)
    fixed, applied = apply_plan(img, p, plan)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
# Every number here is a decision about when a correction earns its cost, so
# each carries the measurement that set it rather than a round number.
SKEW_CORRECT_DEG = 1.0     # measured typical skew on real bundles is 1.2 deg and
                           # costs nothing; below ~1 deg a rotation resamples the
                           # whole page for no gain, so estimate always, rotate rarely
SKEW_MIN_PROMINENCE = 2.5  # ratio of the best profile-variance score to the median
                           # across the sweep; below this the sweep is flat and the
                           # argmax is noise rather than an angle
SKEW_MAX_DEG = 8.0         # beyond this the estimate is more likely wrong than the
                           # page is skewed; report it, refuse to act on it
ILLUM_RANGE = 35.0         # grey levels of spread in the background field. A flat
                           # scan sits near 5-12; a lamp gradient or spine shadow
                           # runs 40+
NOISE_SIGMA = 3.0          # grey levels of true noise, after calibration. A
                           # born-digital render measures 0.00; the scanified
                           # corpus measures 1.6, because degrade.py applies JPEG
                           # last and compression smooths the grain it injected
BLUR_VAR_LOW = 60.0        # variance of Laplacian. Sharp text pages run in the
                           # hundreds; a soft scan drops below ~60
CROP_MIN_KEEP = 0.995      # a content box must contain this fraction of all ink
                           # before it is trusted -- the failure mode of cropping
                           # is amputating a column, which is unrecoverable
CROP_MIN_GAIN = 1.05       # and it must buy at least this much linear resolution,
                           # or it is churn


@dataclass
class Probe:
    """What is measurably true about one page, before anything is changed."""
    width: int
    height: int
    megapixels: float
    illumination_range: float
    noise_sigma: float
    blur_variance: float
    skew_deg: float
    skew_reliable: bool
    ink_fraction: float
    content_box: tuple           # x1, y1, x2, y2
    content_fraction: float      # area of the box over the sheet
    crop_keeps_ink: float        # fraction of ink inside the box
    cost_ms: float = 0.0

    def as_dict(self):
        d = asdict(self)
        d["content_box"] = list(self.content_box)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Plan:
    """Which corrections the evidence supports, and why."""
    deskew: bool = False
    background: bool = False
    clahe: bool = False
    denoise: bool = False
    crop: bool = False
    reasons: dict = field(default_factory=dict)

    def any(self) -> bool:
        return any((self.deskew, self.background, self.clahe, self.denoise, self.crop))

    def as_dict(self):
        return {"deskew": self.deskew, "background": self.background,
                "clahe": self.clahe, "denoise": self.denoise, "crop": self.crop,
                "reasons": self.reasons}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _background(gray, div: int = 8):
    """A low-order estimate of the page's illumination field.

    Closing on a downsample removes text and keeps the lighting. Done at 1/8
    scale because the field is low-order by nature and the estimate costs 5.7 ms
    at full resolution against 2.9 ms here for the same answer.
    """
    small = cv2.resize(gray, (max(1, gray.shape[1] // div), max(1, gray.shape[0] // div)),
                       interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k)
    return cv2.resize(closed, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def _ink_mask(gray):
    """Ink as a mask, without committing to a threshold for the recognisers.

    Otsu is acceptable *here* and nowhere else in this file: the mask is used to
    locate ink and measure skew, never handed to a recogniser, so promoting
    speckle costs a slightly larger bounding box rather than a fabricated glyph.
    """
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def _content_box(mask, keep_percentile: float = 60.0, pad: int = 8):
    """The ink bounding box, by density profile rather than extremes.

    A single speckle in a corner defeats a min/max bounding box, which is the
    reason this uses row and column ink counts above a percentile instead. The
    box is padded so a descender at the boundary is not clipped -- the fixture
    bug that cost this project a week was exactly a clipped descender.
    """
    h, w = mask.shape
    rows = (mask > 0).sum(axis=1).astype(np.float32)
    cols = (mask > 0).sum(axis=0).astype(np.float32)
    if rows.max() <= 0:
        return (0, 0, w, h)

    def span(profile):
        thresh = np.percentile(profile[profile > 0], keep_percentile) if (profile > 0).any() else 0
        idx = np.flatnonzero(profile > max(1.0, thresh * 0.10))
        return (int(idx[0]), int(idx[-1] + 1)) if idx.size else (0, len(profile))

    y1, y2 = span(rows)
    x1, x2 = span(cols)
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))


def _skew_deg(mask, limit: float = 4.0):
    """Skew by projection-profile variance, coarse then fine.

    A correctly de-skewed page has text rows that align, which maximises the
    variance of the row-sum profile's first difference. Measured at 8-9 ms on a
    full page; run on a 1/4 downsample it is under 3 ms for the same angle.
    """
    small = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape
    centre = (w / 2.0, h / 2.0)

    def score(angle):
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        prof = (rot > 0).sum(axis=1).astype(np.float32)
        return float(np.var(np.diff(prof)))

    coarse = np.arange(-limit, limit + 0.01, 0.5)
    scores = np.array([score(a) for a in coarse])
    best = float(coarse[int(scores.argmax())])
    fine = max(np.arange(best - 0.5, best + 0.51, 0.1), key=score)
    angle = float(round(fine, 2))
    # Peak prominence: on a page whose skew is inside the searched range there is
    # one angle where the text rows align and the profile variance spikes. On a
    # page rotated far outside it, no angle aligns anything and the sweep is flat,
    # so the argmax is noise. Measured on a fixture rotated by known amounts --
    # peak/median was 4.38 / 3.67 / 4.64 at 0, 1.5 and 3 degrees, against
    # 2.14 / 1.54 / 1.21 / 1.15 at 6, 12, 25 and 45. Without this test a page at
    # 25 degrees reported +2.30 and would have been confidently rotated by 2.3.
    prominence = float(scores.max() / max(float(np.median(scores)), 1e-9))
    # Saturation means the true angle is outside the searched range, and the
    # returned value is a boundary artefact rather than a measurement. Measured:
    # a page rotated 12 degrees estimates as -4.20 with limit=4, which passed
    # the sanity ceiling and would have been "corrected" by 4.2 degrees --
    # leaving the page 7.8 degrees crooked, having resampled every pixel for
    # nothing. A page that far out needs orientation classification, not fine
    # deskew, so the honest answer is to report and decline.
    return angle, bool(abs(angle) < limit - 0.15 and prominence >= SKEW_MIN_PROMINENCE)


# Measurements run at a bounded working size. The first version probed at full
# resolution and cost 262 ms on an 8.7 MP page -- 2.6% of Granite's entire
# per-page budget, for statistics that do not need that many pixels. Skew,
# illumination and noise are all low-order properties; only the content box
# wants real coordinates, and it is computed on the working mask and scaled back.
_PROBE_MAX_PX = 2_000_000


# Immerkaer's kernel: zero response to any locally planar signal, so what comes
# back is noise rather than content.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
# Calibrated by injecting known Gaussian noise into a clean fixture: the
# flat-region restriction costs roughly half the amplitude, measured as
# estimate = 0.53 x true across sigma 1..24 (1->0.78, 2->1.29, 4->2.30, 8->4.39,
# 16->8.61, 24->12.77). Scaling it back means the reported number is in grey
# levels and the threshold can be set in grey levels too.
_NOISE_CAL = 1.0 / 0.53


def _noise_sigma(gray) -> float:
    """Noise in grey levels, from the flat parts of the page only.

    Two estimators were tried and the first was wrong in both directions. The
    standard deviation of a 3x3 median residual measures glyph EDGES, not noise:
    a crisp synthetic fixture with no noise at all scored 26.1 and tripped a
    denoise it did not need. Taking the MAD of the same residual fixed the false
    positive but quantised badly -- injected sigma of 2, 4 and 8 all reported
    1.48, because a median filter tracks the noise it is supposed to measure
    against. Immerkaer's kernel has no such reference problem, and restricting
    it to low-gradient pixels keeps strokes out of the estimate.
    """
    resid = cv2.filter2D(gray.astype(np.float32), -1, _IMMERKAER)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    flat = cv2.magnitude(gx, gy) < np.percentile(cv2.magnitude(gx, gy), 55.0)
    if flat.sum() < 256:
        return 0.0
    raw = float(np.sqrt(np.pi / 2.0) * np.abs(resid[flat]).mean() / 6.0)
    return raw * _NOISE_CAL


def probe_page(img) -> Probe:
    """Measure a page. No pixels are changed and nothing is decided here."""
    t0 = time.time()
    full = _gray(img)
    h, w = full.shape
    scale = min(1.0, (_PROBE_MAX_PX / float(w * h)) ** 0.5)
    gray = (full if scale >= 1.0 else
            cv2.resize(full, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA))
    bg = _background(gray)
    illum = float(np.percentile(bg, 95) - np.percentile(bg, 5))
    noise = _noise_sigma(gray)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mask = _ink_mask(bg_divide(gray, bg))
    skew, skew_ok = _skew_deg(mask)
    ink = float((mask > 0).mean())
    box_s = _content_box(mask)
    total_ink = float((mask > 0).sum()) or 1.0
    inside = float((mask[box_s[1]:box_s[3], box_s[0]:box_s[2]] > 0).sum())
    # Scale the box back to the real page, and clamp: a box measured on a
    # downsample must never round outward past the sheet it came from.
    inv = 1.0 / scale
    box = (max(0, int(box_s[0] * inv)), max(0, int(box_s[1] * inv)),
           min(w, int(round(box_s[2] * inv))), min(h, int(round(box_s[3] * inv))))
    area = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
    return Probe(width=w, height=h, megapixels=(w * h) / 1e6,
                 illumination_range=illum, noise_sigma=noise, blur_variance=blur,
                 skew_deg=skew, skew_reliable=skew_ok, ink_fraction=ink,
                 content_box=box, content_fraction=area,
                 crop_keeps_ink=inside / total_ink,
                 cost_ms=(time.time() - t0) * 1000.0)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def plan_for(p: Probe, allow_geometry: bool = True) -> Plan:
    """Turn measurements into a plan, recording the evidence for each decision.

    The asymmetry is deliberate and is the whole point of the module. A crop
    spends an existing pixel budget better and cannot invent anything, so it
    runs whenever it is safe. A rotation resamples every pixel on the page, so
    it waits for evidence that the page is actually crooked.
    """
    plan = Plan()
    r = plan.reasons

    gain = (1.0 / p.content_fraction) ** 0.5 if p.content_fraction > 0 else 1.0
    if p.crop_keeps_ink >= CROP_MIN_KEEP and gain >= CROP_MIN_GAIN:
        plan.crop = True
        r["crop"] = (f"content is {p.content_fraction*100:.0f}% of the sheet -> "
                     f"{gain:.2f}x linear resolution for the same pixel budget, "
                     f"keeping {p.crop_keeps_ink*100:.2f}% of ink")
    else:
        r["crop"] = (f"declined: keeps {p.crop_keeps_ink*100:.2f}% of ink "
                     f"(need {CROP_MIN_KEEP*100:.1f}%), gain {gain:.2f}x "
                     f"(need {CROP_MIN_GAIN:.2f}x)")

    if not allow_geometry:
        r["deskew"] = "declined: geometry changes disabled for this run"
    elif not p.skew_reliable:
        r["deskew"] = (f"declined: the skew estimate ({p.skew_deg:+.2f} deg) is not "
                       f"trustworthy -- it either saturated the +-4 deg search or the "
                       f"sweep had no clear peak, both of which mean the page is "
                       f"rotated further than fine deskew can measure. That needs "
                       f"orientation classification; flagged for review instead")
    elif abs(p.skew_deg) > SKEW_MAX_DEG:
        r["deskew"] = (f"declined: estimate {p.skew_deg:+.2f} deg exceeds "
                       f"{SKEW_MAX_DEG} deg, more likely a bad estimate than a "
                       f"bad page -- reported, not corrected")
    elif abs(p.skew_deg) >= SKEW_CORRECT_DEG:
        plan.deskew = True
        r["deskew"] = f"{p.skew_deg:+.2f} deg >= {SKEW_CORRECT_DEG} deg"
    else:
        r["deskew"] = (f"declined: {p.skew_deg:+.2f} deg is below "
                       f"{SKEW_CORRECT_DEG} deg, not worth resampling the page")

    if p.illumination_range >= ILLUM_RANGE:
        plan.background = plan.clahe = True
        r["background"] = (f"illumination spread {p.illumination_range:.1f} grey "
                           f"levels >= {ILLUM_RANGE}")
    else:
        r["background"] = (f"declined: illumination spread "
                           f"{p.illumination_range:.1f} < {ILLUM_RANGE}, the page "
                           f"is already evenly lit")

    if p.noise_sigma >= NOISE_SIGMA:
        plan.denoise = True
        r["denoise"] = f"noise sigma {p.noise_sigma:.1f} >= {NOISE_SIGMA} (3x3 median only)"
    else:
        r["denoise"] = f"declined: noise sigma {p.noise_sigma:.1f} < {NOISE_SIGMA}"

    if p.blur_variance < BLUR_VAR_LOW:
        r["blur"] = (f"page is soft (Laplacian variance {p.blur_variance:.0f} < "
                     f"{BLUR_VAR_LOW}); recorded for review, NOT sharpened -- "
                     f"deconvolution costs ~215 ms/page and amplifies the noise "
                     f"it cannot distinguish from stroke")
    return plan


# --------------------------------------------------------------------------- #
# corrections
# --------------------------------------------------------------------------- #
def bg_divide(gray, bg=None):
    """Flatten the illumination field by division. Grayscale in, grayscale out."""
    bg = _background(gray) if bg is None else bg
    safe = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip(gray.astype(np.float32) / safe * 255.0, 0, 255).astype(np.uint8)


def apply_plan(img, p: Probe, plan: Plan):
    """Apply the planned corrections, in the order a scanner's damage occurred.

    Reverse of degrade.py's pipeline: photometry is undone before geometry,
    because the illumination field is fixed to the platen and does not rotate
    with the page. The crop comes last so it measures the corrected page.

    Returns (image, applied) where applied lists what actually ran, so the audit
    records the treatment rather than the intention.
    """
    applied, colour = [], img.ndim == 3
    work = _gray(img)

    if plan.background:
        work = bg_divide(work)
        applied.append("background-division")
    if plan.denoise:
        work = cv2.medianBlur(work, 3)
        applied.append("median3")
    if plan.clahe:
        work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)
        applied.append("clahe")
    if plan.deskew:
        h, w = work.shape
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), p.skew_deg, 1.0)
        # CUBIC deliberately: measured 39 ms against LANCZOS4's 202 ms on a
        # 2481x3508 page, for no measurable difference in what the OCR reads.
        # White border, because rotating black into a page margin creates an
        # edge that the layout detector reads as a rule.
        work = cv2.warpAffine(work, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        applied.append(f"deskew{p.skew_deg:+.2f}deg")
    if plan.crop:
        x1, y1, x2, y2 = p.content_box
        if x2 - x1 > 16 and y2 - y1 > 16:
            work = work[y1:y2, x1:x2]
            applied.append(f"crop{x2-x1}x{y2-y1}")

    out = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR) if colour else work
    return out, applied


def prepare(img, allow_geometry: bool = True):
    """Probe, plan and apply in one call. Returns (image, probe, plan, applied)."""
    p = probe_page(img)
    plan = plan_for(p, allow_geometry=allow_geometry)
    out, applied = apply_plan(img, p, plan) if plan.any() else (img, [])
    return out, p, plan, applied
