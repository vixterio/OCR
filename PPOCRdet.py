"""Text detection only — PP-OCRv6 medium detection model.

Detection finds *where* the text is: it returns a quadrilateral polygon per
text line and no characters. Use PPOCRrec.py to read the characters, or
PaddleOCR (the full pipeline) to do both in one pass.

    ./safe_run.sh PPOCRdet.py demo.png
"""
import os
import sys

from paddleocr import TextDetection

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "demo.png"

detector = TextDetection(model_name="PP-OCRv6_medium_det")

os.makedirs("output/det", exist_ok=True)
for res in detector.predict(IMAGE):
    res.print()
    res.save_to_img("output/det")   # source image with boxes drawn on
    res.save_to_json("output/det")  # polygons + confidence scores
