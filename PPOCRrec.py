"""Text recognition only — PP-OCRv6 medium recognition model.

Recognition reads characters out of an image that has *already* been cropped to
a single text line. It does no detection, so handing it a full page returns one
garbled string for the whole thing. Run PPOCRdet.py first to get the boxes, or
use PaddleOCR (the full pipeline) to chain both.

    ./safe_run.sh PPOCRrec.py line_crop.png

With no argument it crops the first detected line out of demo.png so the script
is runnable on its own.
"""
import os
import sys

import cv2
from paddleocr import TextDetection, TextRecognition


def first_detected_line(image_path, out_path):
    """Crop the first text line PP-OCRv6 detection finds — gives rec a valid input."""
    det = TextDetection(model_name="PP-OCRv6_medium_det")
    res = next(iter(det.predict(image_path)))
    poly = res["dt_polys"][0]
    xs, ys = [int(p[0]) for p in poly], [int(p[1]) for p in poly]
    img = cv2.imread(image_path)
    cv2.imwrite(out_path, img[min(ys) : max(ys), min(xs) : max(xs)])
    return out_path


if len(sys.argv) > 1:
    image = sys.argv[1]
else:
    os.makedirs("output/rec", exist_ok=True)
    image = first_detected_line("demo.png", "output/rec/line_crop.png")
    print(f"no image given; using first detected line -> {image}", flush=True)

recognizer = TextRecognition(model_name="PP-OCRv6_medium_rec")

os.makedirs("output/rec", exist_ok=True)
for res in recognizer.predict(image):
    res.print()                     # {'rec_text': ..., 'rec_score': ...}
    res.save_to_json("output/rec")
