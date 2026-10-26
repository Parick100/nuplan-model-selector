"""Page background plates: light cream + baked geometry (shapes do not render in this PPTD build)."""
import os
from PIL import Image, ImageDraw

OUT = os.path.join("pre1_ppt", "bg")
os.makedirs(OUT, exist_ok=True)

S = 2
W, H = 960 * S, 540 * S

CREAM = "#F1EDE3"
CREAM2 = "#EAE5D9"
RULE = "#CFC8B8"
ORANGE = "#E85D26"
SOFTBOX = "#D8D2C4"
BAR = "#C4BDAE"


def rect(d, x, y, w, h, color):
    d.rectangle([x * S, y * S, (x + w) * S - 1, (y + h) * S - 1], fill=color)


def rule(d, x, y, w, color, t=1):
    d.rectangle([x * S, y * S, (x + w) * S - 1, y * S + t * S - 1], fill=color)


def plate(bg, name, hairline=True):
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    if hairline:
        rule(d, 48, 66, 864, RULE, 1)
    return im, d


def save(im, name):
    p = os.path.join(OUT, name)
    im.save(p)
    print("wrote", name, os.path.getsize(p) // 1024, "KB")


# p01 cover — orange accent bar under the title
im, d = plate(CREAM, "p01", hairline=False)
rect(d, 48, 392, 180, 6, ORANGE)
save(im, "p01.png")

# p02 problem
save(plate(CREAM, "p02")[0], "p02.png")

# p03 the two methods
save(plate(CREAM, "p03")[0], "p03.png")

# p04 how it runs on nuPlan
save(plate(CREAM, "p04")[0], "p04.png")

# p05 the data
save(plate(CREAM, "p05")[0], "p05.png")

# p06 the question (chart)
save(plate(CREAM, "p06")[0], "p06.png")

# p07 the plan — pipeline boxes + metric hairlines
im, d = plate(CREAM, "p07")
for i in range(5):
    x = 60 + i * 172
    rect(d, x, 138, 152, 46, ORANGE if i == 4 else SOFTBOX)
for y in (290, 334, 378, 422):
    rule(d, 48, y, 864, RULE, 1)
save(im, "p07.png")

# p08 the decision table — rules
im, d = plate(CREAM, "p08")
rule(d, 48, 158, 864, RULE, 1)
for y in (216, 258, 300, 342):
    rule(d, 48, y, 864, "#DCD6C8", 1)
rule(d, 48, 384, 864, "#26231E", 2)
save(im, "p08.png")

# p09 the goal — selector figure carries it
save(plate(CREAM, "p09")[0], "p09.png")

# p10 team — gantt bars + gate rule
im, d = plate(CREAM, "p10")
rect(d, 300, 176, 120, 20, ORANGE)
rect(d, 300, 232, 360, 20, BAR)
rect(d, 420, 288, 360, 20, BAR)
rect(d, 540, 344, 360, 20, BAR)
rule(d, 48, 398, 864, RULE, 1)
save(im, "p10.png")

# p11 closing
im, d = plate(CREAM, "p11", hairline=False)
rect(d, 48, 392, 180, 6, ORANGE)
save(im, "p11.png")

print("done")
