"""Regenerate the p08 / p09 / p10 plates for the revised pages."""
import os
from PIL import Image, ImageDraw

OUT = os.path.join("pre1_ppt", "bg")
os.makedirs(OUT, exist_ok=True)
S = 2
W, H = 960 * S, 540 * S
CREAM = "#F1EDE3"
RULE = "#CFC8B8"
ORANGE = "#E85D26"
BAR = "#C4BDAE"


def rect(d, x, y, w, h, color):
    d.rectangle([x * S, y * S, (x + w) * S - 1, (y + h) * S - 1], fill=color)


def rule(d, x, y, w, color, t=1):
    d.rectangle([x * S, y * S, (x + w) * S - 1, y * S + t * S - 1], fill=color)


def plate(name, hairline=True):
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    if hairline:
        rule(d, 48, 66, 864, RULE, 1)
    return im, d


def save(im, name):
    p = os.path.join(OUT, name)
    im.save(p)
    print("wrote", name, os.path.getsize(p) // 1024, "KB")


# p08 — the selector page, no table rules any more
save(plate("p08")[0], "p08.png")

# p09 — the payoff page
save(plate("p09")[0], "p09.png")

# p10 — re-planned gantt bars + gate rule
im, d = plate("p10")
rect(d, 300, 176, 120, 20, ORANGE)   # A  W1
rect(d, 300, 232, 480, 20, BAR)      # B  W1-W4
rect(d, 420, 288, 360, 20, BAR)      # C  W2-W4
rect(d, 420, 344, 480, 20, BAR)      # D  W2-W5
rule(d, 48, 398, 864, RULE, 1)
save(im, "p10.png")

print("done")
