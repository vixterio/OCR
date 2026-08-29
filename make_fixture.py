"""Render the European test table at a chosen scale.

Scale stands in for rasterisation DPI: drawing the same layout at 2x means every
glyph gets 4x the pixels, exactly as re-rendering a PDF at 144 instead of 72 DPI
would. Used to measure what resolution buys and what it costs.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
out = sys.argv[2] if len(sys.argv) > 2 else f"euro_{int(S)}x.png"
F = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
W, H = int(1060 * S), int(510 * S)
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
fb = ImageFont.truetype(F, int(28 * S))
f = ImageFont.truetype(F, int(25 * S))
d.text((40 * S, 22 * S), "Sprawozdanie finansowe / Οικονομική έκθεση", fill="black", font=fb)
rows = [("Region", "Kwartał 1", "Kwartał 2", "Razem"),
        ("Zażółć", "1.284,50", "2 019,75", "3 304,25"),
        ("Gęślą", "987,31", "1.556,40", "2 543,71"),
        ("Αθήνα", "3.412,66", "1 073,82", "4 486,48"),
        ("Κρήτη", "845,09", "2.764,53", "3 609,62")]
y = 80 * S
for i, r in enumerate(rows):
    x = 40 * S
    for c, cell in enumerate(r):
        d.text((x, y), cell, fill="black", font=(fb if i == 0 else f))
        x += (230 if c == 0 else 270) * S
    y += 72 * S
    d.line([(40 * S, y - 14 * S), (W - 40 * S, y - 14 * S)], fill="black", width=max(1, int(S)))
d.text((40 * S, y + 8 * S), "Wzrost 18,7% — Ανάπτυξη 42,3%", fill="black", font=f)
img.save(out)
print(f"{out} {img.size}")
