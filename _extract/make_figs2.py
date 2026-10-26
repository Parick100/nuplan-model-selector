"""Rebuild all figures: light editorial palette + Times New Roman."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches

OUT = os.path.join("pre1_ppt", "assets")
os.makedirs(OUT, exist_ok=True)

CREAM = "#F1EDE3"
CREAM2 = "#E8E3D7"
CARD = "#FBF9F4"
INK = "#26231E"
INK2 = "#6B6559"
RULE = "#CFC8B8"
ORANGE = "#E85D26"
ORANGE_L = "#EFB08C"
ROAD = "#E0DACB"
BLOCK = "#E7E1D3"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name, os.path.getsize(p) // 1024, "KB")


# ------------------------------------------------------- F1: the averaging problem
def f1_problem():
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 20), 100, 13, color=ROAD, zorder=1))
    ax.add_patch(Rectangle((43, 0), 14, 52, color=ROAD, zorder=1))
    for (x, y, w, h) in [(0, 33, 43, 19), (57, 33, 43, 19), (0, 0, 43, 20), (57, 0, 43, 20)]:
        ax.add_patch(Rectangle((x, y), w, h, color=BLOCK, zorder=1))

    for xs, ys in [((0, 43), (26.5, 26.5)), ((57, 100), (26.5, 26.5)),
                   ((50, 50), (0, 20)), ((50, 50), (33, 52))]:
        ax.plot(list(xs), list(ys), color=RULE, lw=1.1, ls=(0, (7, 6)), zorder=2)

    cx, cy = 50, 14
    ax.add_patch(FancyBboxPatch((cx - 1.5, cy - 2.4), 3.0, 4.8,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                fc=INK, ec="none", zorder=4))
    ax.text(cx - 2.8, cy - 0.4, "history", color=INK2, fontsize=11,
            ha="right", va="center", zorder=6, style="italic")

    def curve(pts, color, lw=3.2, z=6):
        p = Path(pts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(mpatches.PathPatch(p, fc="none", ec=color, lw=lw, zorder=z,
                                        capstyle="round"))

    curve([(50, 15), (50, 26), (50, 34), (50, 46)], INK)
    ax.text(50, 47.4, "straight", color=INK, fontsize=12, ha="center", va="bottom")
    curve([(50, 15), (50, 28), (40, 34), (24, 34)], ORANGE)
    ax.text(22.5, 34, "left", color=ORANGE, fontsize=12, ha="right", va="center")
    curve([(50, 15), (50, 26), (62, 28), (78, 28)], "#B9A88F")
    ax.text(79.5, 28, "right", color="#9A8B72", fontsize=12, ha="left", va="center")

    ax.add_patch(FancyArrowPatch((50, 15), (33.5, 40.5), arrowstyle="-", lw=2.8,
                                 ls=(0, (5, 3.5)), color="#8C8574", zorder=7))
    ax.plot([32.2, 35.6], [39.2, 42.6], color="#8C8574", lw=2.8, zorder=8, solid_capstyle="round")
    ax.plot([32.2, 35.6], [42.6, 39.2], color="#8C8574", lw=2.8, zorder=8, solid_capstyle="round")

    ax.text(7.0, 46.6, "the average", color="#7A7364", fontsize=15,
            ha="left", va="bottom", fontweight="bold", zorder=9)
    ax.text(7.0, 43.7, "drives into the kerb", color="#7A7364", fontsize=11.5,
            ha="left", va="bottom", zorder=9, style="italic")

    save(fig, "f1_problem.png")


# ------------------------------------------------------- F2: how the two methods work
def f2_methods():
    fig = plt.figure(figsize=(9.8, 4.6), facecolor=CREAM)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.07,
                          left=0.015, right=0.985, top=0.99, bottom=0.02)

    # ---- panel A: diffusion = learn to undo a fixed noising process
    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(CREAM)
    axA.set_xlim(0, 100)
    axA.set_ylim(0, 78)
    axA.axis("off")

    rng = np.random.default_rng(11)
    ncards = 5
    w, gap = 17.0, 3.4
    x0 = (100 - (ncards * w + (ncards - 1) * gap)) / 2
    xs = [x0 + i * (w + gap) for i in range(ncards)]

    tt = np.linspace(0, 1, 120)
    traj_x = 3.0 + 11.0 * tt
    traj_y = 3.0 + 12.0 * tt ** 0.75 + 2.2 * np.sin(np.pi * tt * 1.15)

    CY, CH = 26.0, 32.0
    for i, x in enumerate(xs):
        axA.add_patch(FancyBboxPatch((x, CY), w, CH,
                                     boxstyle="round,pad=0.2,rounding_size=1.2",
                                     fc=CARD, ec=RULE, lw=0.9, zorder=1))
        alpha = [1.0, 0.62, 0.30, 0.10, 0.0][i]
        if alpha > 0:
            axA.plot(x + traj_x * w / 17.0, CY + traj_y * CH / 17.0,
                     color=INK, lw=1.9, alpha=alpha, zorder=3, solid_capstyle="round")
        ndots = [0, 160, 460, 950, 1500][i]
        if ndots:
            px = x + rng.uniform(0.6, w - 0.6, ndots)
            py = CY + rng.uniform(0.6, CH - 0.6, ndots)
            axA.scatter(px, py, s=1.6, color=["", "#9A9384", "#8C8574", "#7A7364", "#6B6559"][i],
                        alpha=0.55, zorder=2, linewidths=0)
        axA.text(x + w / 2, CY - 5.0, ["a real future", "", "", "", "pure noise"][i],
                 color=INK2, fontsize=9.5, ha="center", va="center")
        if i in (0, 4):
            axA.text(x + w / 2, CY - 10.0, ["$x_0$", "$x_T$"][0 if i == 0 else 1],
                     color=INK, fontsize=14, ha="center", va="center")

    axA.add_patch(FancyArrowPatch((x0, 64.0), (x0 + ncards * w + 4 * gap, 64.0),
                                  arrowstyle="-|>", mutation_scale=13, lw=1.4, color=INK2))
    axA.text(50, 65.4, "add noise   —   how it is trained", color=INK2, fontsize=11.5,
             ha="center", va="bottom")
    axA.add_patch(FancyArrowPatch((x0 + ncards * w + 4 * gap, 10.0), (x0, 10.0),
                                  arrowstyle="-|>", mutation_scale=13, lw=1.4, color=ORANGE))
    axA.text(50, 4.5, "remove noise   —   how it samples", color=ORANGE, fontsize=11.5,
             ha="center", va="bottom", fontweight="bold")

    axA.text(50, 77.5, "DIFFUSION", color=INK, fontsize=16,
             ha="center", va="top", fontweight="bold")
    axA.text(50, 71.8, "learn to undo a fixed noising process", color=INK2, fontsize=10.5,
             ha="center", va="top", style="italic")

    # ---- panel B: flow matching = learn a flow onto the data
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(CREAM)
    axB.set_xlim(0, 100)
    axB.set_ylim(0, 78)
    axB.axis("off")

    cx, cy = 22, 36
    for r, a in zip(np.linspace(3.5, 14, 9), np.linspace(0.95, 0.16, 9)):
        axB.add_patch(Circle((cx, cy), r, fc="none", ec=ORANGE, lw=1.5, alpha=a, zorder=3))
    axB.add_patch(Circle((cx, cy), 2.6, fc=ORANGE, ec="none", alpha=0.85, zorder=4))
    axB.text(cx, 15.0, "noise", color=ORANGE, fontsize=13.5, ha="center", va="center",
             fontweight="bold")

    rng2 = np.random.default_rng(5)
    lobe1 = rng2.normal([74, 44], [4.6, 3.6], size=(210, 2))
    lobe2 = rng2.normal([79, 27], [4.0, 3.2], size=(210, 2))
    pts = np.vstack([lobe1, lobe2])
    axB.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, alpha=0.6, zorder=4, linewidths=0)

    starts = []
    for ang in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        rr = 5.0 + 9.0 * rng2.random()
        starts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
    ends = pts[rng2.choice(len(pts), len(starts), replace=False)]

    t = np.linspace(0, 1, 70)
    for (sx, sy), (ex, ey) in zip(starts, ends):
        bend = 0.16 * (ey - sy)
        bx = (1 - t) * sx + t * ex
        by = (1 - t) * sy + t * ey + bend * np.sin(np.pi * t)
        axB.plot(bx, by, color=ORANGE, lw=1.15, alpha=0.42, zorder=2)

    axB.text(76, 62.0, "real futures", color=INK, fontsize=13.5, ha="center", va="center",
             fontweight="bold")
    axB.text(50, 77.5, "FLOW MATCHING", color=INK, fontsize=16,
             ha="center", va="top", fontweight="bold")
    axB.text(50, 71.8, "learn a flow that carries noise onto real data",
             color=INK2, fontsize=10.5, ha="center", va="top", style="italic")

    save(fig, "f2_methods.png")


# ------------------------------------------------------- F3: accuracy vs compute
def f3_budget():
    fig, ax = plt.subplots(figsize=(9.0, 4.7), facecolor=CREAM)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.17)
    ax.set_facecolor(CREAM)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=11, length=0, pad=6)
    ax.set_xscale("log", base=2)

    c = np.geomspace(2, 256, 200)
    regression = np.full_like(c, 1.62)
    flow = 0.68 + 1.42 * np.exp(-0.72 * np.log2(c))
    diffusion = 0.55 + 1.15 * np.exp(-0.32 * np.log2(c))

    l1, = ax.plot(c, regression, color="#8C8574", lw=2.6, ls=(0, (5, 4)), label="regression")
    l2, = ax.plot(c, flow, color=INK, lw=3.0, label="flow matching")
    l3, = ax.plot(c, diffusion, color=ORANGE, lw=3.0, label="diffusion")
    ax.scatter([2], [regression[0]], s=48, color="#8C8574", zorder=5)
    ax.axvline(95, color=RULE, lw=1.2, ls=(0, (2, 3)), zorder=0)

    ax.set_xlim(1.55, 420)
    ax.set_ylim(0.35, 2.55)
    ax.set_xticks([2, 4, 16, 64, 256])
    ax.set_xticklabels(["2", "4", "16", "64", "256"], color=INK2)
    ax.set_yticks([])
    ax.grid(axis="x", color=RULE, lw=0.9, alpha=0.6, zorder=0)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=13,
                    handlelength=1.7, handletextpad=0.7, borderpad=0.0, labelspacing=0.7)
    for t, ln in zip(leg.get_texts(), [l1, l2, l3]):
        t.set_color(ln.get_color())

    ax.annotate("crossover", xy=(95, 0.72), xytext=(128, 1.10), color=INK2,
                fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=RULE, lw=1.1))

    fig.text(0.05, 0.045, "error  —  lower is better", color=INK2, fontsize=11,
             ha="left", va="center", style="italic")
    fig.text(0.97, 0.045, "test-time compute  —  number of model calls",
             color=INK2, fontsize=11, ha="right", va="center", style="italic")

    save(fig, "f3_budget.png")


# ------------------------------------------------------- F4: how it runs on nuPlan
def f4_arch():
    fig, ax = plt.subplots(figsize=(9.8, 3.9), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def card(x, y, w, h, fc=CARD, ec=RULE, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=1.3",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.3, color=INK2, zorder=3))

    Y, H = 9.0, 22.0

    # scene
    card(0.5, Y, 15, H)
    ax.add_patch(Rectangle((1.6, Y + 1.4), 12.8, H - 2.8, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((1.6, Y + H / 2 - 1.6), 12.8, 3.2, color=ROAD, zorder=3))
    for (ax_, ay_) in [(3.0, Y + 4.0), (6.0, Y + 14.0), (9.5, Y + 6.0), (11.8, Y + 16.0)]:
        ax.add_patch(Rectangle((ax_, ay_), 1.5, 2.6, color="#B9B2A2", zorder=4))
    ax.add_patch(Rectangle((7.2, Y + H / 2 - 1.3), 1.5, 2.6, color=ORANGE, zorder=5))
    ax.text(8.0, Y - 3.2, "SCENE", color=INK, fontsize=12.5, ha="center",
            va="center", fontweight="bold")

    arrow(16.0, 20.0, Y + H / 2)

    # inputs
    names = ["NEIGHBOURS", "LANES", "STATIC OBJECTS", "NAVIGATION"]
    ih = 4.6
    for i, nm in enumerate(names):
        yy = Y + H - (i + 1) * ih - i * 0.9
        card(20.5, yy, 17.5, ih, fc=CARD)
        ax.text(29.25, yy + ih / 2, nm, color=INK2, fontsize=8.6, ha="center", va="center")
    ax.text(29.25, Y + H + 2.6, "WHAT WE FEED IT", color=INK, fontsize=11,
            ha="center", va="bottom", fontweight="bold")

    arrow(38.5, 42.5, Y + H / 2)

    # encoder
    card(43.0, Y, 15, H)
    ax.text(50.5, Y + H / 2 + 3.4, "ENCODER", color=INK, fontsize=12, ha="center",
            va="center", fontweight="bold")
    ax.text(50.5, Y + H / 2 - 3.0, "MLP-Mixer\n+ self-attention", color=INK2, fontsize=8.8,
            ha="center", va="center")

    arrow(58.5, 62.5, Y + H / 2)

    # decoder
    card(63.0, Y, 19, H, fc="#F7E9DF", ec=ORANGE_L)
    ax.text(72.5, Y + H / 2 + 4.0, "DECODER", color=INK, fontsize=12, ha="center",
            va="center", fontweight="bold")
    ax.text(72.5, Y + H / 2 - 2.4, "diffusion transformer\ndenoiser", color=INK2,
            fontsize=8.8, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((69.5, Y + 2.6), (75.5, Y + 2.6),
                                 connectionstyle="arc3,rad=-1.6", arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color=ORANGE, zorder=4))
    ax.text(72.5, Y + H + 2.6, "ONLY THIS DIFFERS", color=ORANGE, fontsize=11,
            ha="center", va="bottom", fontweight="bold")

    arrow(82.5, 86.5, Y + H / 2)

    # output
    card(87.0, Y, 12.5, H)
    ax.add_patch(Rectangle((88.0, Y + 1.4), 10.5, H - 2.8, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((88.0, Y + H / 2 - 1.6), 10.5, 3.2, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((89.4, Y + H / 2 - 1.3), 1.5, 2.6, color=ORANGE, zorder=5))
    tx = np.linspace(0, 1, 60)
    ax.plot(91.2 + 5.6 * tx, Y + H / 2 + 9.0 * tx ** 0.8, color=INK, lw=1.8, zorder=5)
    ax.text(93.25, Y - 3.2, "FUTURE", color=INK, fontsize=12.5, ha="center",
            va="center", fontweight="bold")

    save(fig, "f4_arch.png")


# ------------------------------------------------------- F5: the selector we aim to build
def f5_selector():
    fig, ax = plt.subplots(figsize=(9.4, 3.3), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def card(x, y, w, h, fc=CARD, ec=RULE, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.3, color=INK2, zorder=3))

    Y, H = 12.0, 16.0
    card(1, Y, 19, H)
    ax.text(10.5, Y + H / 2 + 2.6, "READ THE SCENE", color=INK, fontsize=11,
            ha="center", va="center", fontweight="bold")
    ax.text(10.5, Y + H / 2 - 3.0, "junction? budget?\ncan we score?", color=INK2,
            fontsize=9, ha="center", va="center")
    arrow(20.5, 25.5, Y + H / 2)

    card(26, Y - 1.5, 21, H + 3, fc="#F7E9DF", ec=ORANGE, lw=1.5)
    ax.text(36.5, Y + H / 2 + 3.2, "THE SELECTOR", color=ORANGE, fontsize=12.5,
            ha="center", va="center", fontweight="bold")
    ax.text(36.5, Y + H / 2 - 3.4, "picks the model", color=INK2, fontsize=9,
            ha="center", va="center")
    arrow(47.5, 52.5, Y + H / 2)

    card(53, Y, 19, H)
    ax.text(62.5, Y + H / 2 + 2.6, "ONE MODEL", color=INK, fontsize=11,
            ha="center", va="center", fontweight="bold")
    ax.text(62.5, Y + H / 2 - 3.0, "runs and returns\na future", color=INK2,
            fontsize=9, ha="center", va="center")
    arrow(72.5, 77.5, Y + H / 2)

    card(78, Y, 21, H, fc=CARD, ec=INK, lw=1.3)
    ax.text(88.5, Y + H / 2 + 2.6, "nuPLAN SCORER", color=INK, fontsize=11,
            ha="center", va="center", fontweight="bold")
    ax.text(88.5, Y + H / 2 - 3.0, "the official score", color=INK2,
            fontsize=9, ha="center", va="center")

    ax.add_patch(FancyArrowPatch((36.5, Y - 1.5), (36.5, 5.5), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color=ORANGE,
                                 connectionstyle="arc3,rad=0.0"))
    ax.text(38.5, 3.4, "the question:  does a mixed policy beat every single model?",
            color=ORANGE, fontsize=11, ha="left", va="center", fontweight="bold")

    save(fig, "f5_selector.png")


if __name__ == "__main__":
    f1_problem()
    f2_methods()
    f3_budget()
    f4_arch()
    f5_selector()
