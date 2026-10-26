"""Selector figures: the routing mechanism, and the quality-vs-decision-time payoff."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = os.path.join("pre1_ppt", "assets")
os.makedirs(OUT, exist_ok=True)

CREAM = "#F1EDE3"
CARD = "#FBF9F4"
INK = "#26231E"
INK2 = "#6B6559"
RULE = "#CFC8B8"
ORANGE = "#E85D26"
SOFT = "#F7E9DF"
ROAD = "#E0DACB"

plt.rcParams["font.family"] = "Times New Roman"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name, os.path.getsize(p) // 1024, "KB")


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=%s" % r,
                                fc=fc, ec=ec, lw=lw, zorder=2))


def arw(ax, x1, x2, y, color=INK2, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=color, zorder=3))


# ------------------------------------------------- F6: the routing mechanism
def f6_routing():
    fig, ax = plt.subplots(figsize=(9.8, 3.3), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    # the scene
    rbox(ax, 2, 6, 15, 22, CARD, RULE)
    ax.add_patch(Rectangle((3.1, 7.2), 12.8, 19.6, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((3.1, 15.6), 12.8, 3.2, color=ROAD, zorder=3))
    for (px, py) in [(4.2, 8.6), (7.0, 21.0), (11.0, 9.8), (13.2, 20.0)]:
        ax.add_patch(Rectangle((px, py), 1.5, 2.4, color="#B9B2A2", zorder=4))
    ax.add_patch(Rectangle((8.1, 15.9), 1.5, 2.4, color=ORANGE, zorder=5))
    ax.text(9.5, 3.0, "THE SCENE", color=INK, fontsize=11.5, ha="center",
            va="center", fontweight="bold")

    arw(ax, 17.8, 21.6, 17)

    # the selector
    rbox(ax, 22, 9, 17, 16, SOFT, ORANGE, lw=1.6)
    ax.text(30.5, 21.0, "SELECTOR", color=ORANGE, fontsize=12.5, ha="center",
            va="center", fontweight="bold")
    ax.text(30.5, 17.4, "reads the scene,\npicks one model", color=INK2, fontsize=8.8,
            ha="center", va="center")
    ax.text(30.5, 13.0, "cost \u2248 noise", color=ORANGE, fontsize=9.5, ha="center",
            va="center", fontweight="bold")
    ax.text(30.5, 3.0, "THE SELECTOR", color=INK, fontsize=11.5, ha="center",
            va="center", fontweight="bold")

    arw(ax, 39.6, 43.4, 17)

    # the models it can choose from
    models = [("regression", "1\u00d7", 21.5, False),
              ("flow matching", "4\u00d7", 12.6, True),
              ("diffusion", "15\u00d7", 3.7, False)]
    for name, cost, y, chosen in models:
        rbox(ax, 44, y, 23, 7.6, CARD, ORANGE if chosen else RULE,
             lw=1.7 if chosen else 1.0)
        ax.text(45.6, y + 3.8, name, color=INK if chosen else INK2, fontsize=10.5,
                ha="left", va="center", fontweight="bold" if chosen else "normal")
        ax.text(65.5, y + 3.8, cost, color=ORANGE if chosen else INK2, fontsize=10.5,
                ha="right", va="center", fontweight="bold")
    ax.text(55.5, 31.0, "runs exactly one", color=INK2, fontsize=9.5, ha="center",
            va="center", style="italic")

    arw(ax, 67.6, 71.4, 16.4, color=ORANGE, lw=1.8)

    # the answer
    rbox(ax, 72, 6, 26, 22, CARD, RULE)
    ax.add_patch(Rectangle((73.2, 7.2), 23.6, 19.6, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((73.2, 15.6), 23.6, 3.2, color=ROAD, zorder=3))
    ax.add_patch(Rectangle((75.0, 15.9), 1.6, 2.4, color=ORANGE, zorder=5))
    tt = np.linspace(0, 1, 60)
    ax.plot(78.5 + 15.0 * tt, 17.1 + 8.6 * tt ** 0.8, color=INK, lw=1.9, zorder=5)
    ax.text(85.0, 3.0, "A FUTURE", color=INK, fontsize=11.5, ha="center",
            va="center", fontweight="bold")

    save(fig, "f6_routing.png")


# ------------------------------------------------- F7: quality vs decision time
def f7_payoff():
    fig, ax = plt.subplots(figsize=(9.2, 4.5), facecolor=CREAM)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.19)
    ax.set_facecolor(CREAM)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=11, length=0, pad=6)

    pts = {
        "regression":   (1.0, 1.62, INK2, False),
        "flow matching": (4.0, 0.95, INK, False),
        "selector":     (5.5, 0.62, ORANGE, True),
        "diffusion":    (15.0, 0.58, INK, False),
    }

    ax.plot([4.0, 15.0], [0.95, 0.58], color=RULE, lw=1.6, ls=(0, (6, 4)), zorder=1)
    ax.text(9.6, 0.905, "any single fixed model", color=INK2, fontsize=10.5,
            ha="center", va="bottom", style="italic")

    for name, (x, y, c, big) in pts.items():
        ax.scatter([x], [y], s=180 if big else 95, color=c, zorder=5,
                   edgecolor=CREAM, linewidth=1.2)
        if name == "regression":
            ha, dy = "left", 0.05
        elif name == "diffusion":
            ha, dy = "right", 0.05
        else:
            ha, dy = "center", 0.06
        ax.text(x, y - dy, name, color=c, fontsize=12.5, ha=ha, va="top",
                fontweight="bold" if big else "normal", zorder=6)

    # the gap the selector buys at the same decision time
    ax.annotate("", xy=(5.5, 0.635), xytext=(5.5, 0.885),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.5,
                                mutation_scale=11))
    ax.text(5.95, 0.762, "the gap the selector buys", color=ORANGE, fontsize=11,
            ha="left", va="center", fontweight="bold")

    ax.set_xlim(0.2, 17.2)
    ax.set_ylim(0.34, 1.92)
    ax.set_xticks([1, 4, 8, 12, 15])
    ax.set_xticklabels(["1\u00d7", "4\u00d7", "8\u00d7", "12\u00d7", "15\u00d7"], color=INK2)
    ax.set_yticks([])
    ax.grid(axis="x", color=RULE, lw=0.9, alpha=0.55, zorder=0)

    ax.text(0.35, 1.86, "error  \u2014  lower is better", color=INK2, fontsize=11,
            ha="left", va="top", style="italic")
    ax.text(17.1, 1.86, "average decision time per prediction", color=INK2,
            fontsize=11, ha="right", va="top", style="italic")

    save(fig, "f7_payoff.png")


if __name__ == "__main__":
    f6_routing()
    f7_payoff()
