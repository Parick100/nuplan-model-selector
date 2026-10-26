"""为 PPT 准备的图包：一次前向算出所有需要的量，输出 7 张独立 PNG。

每张图都是**独立文件**，方便直接往幻灯片里塞。全部英文标注（matplotlib 默认字体
没有中文字形，中文会渲染成方块）。

    python code/runners/make_figs.py --suffix _lp2 --out-dir /path/to/figs

产出：
    fig01_data.png     数据概览（速度/转弯/终点距离分布 + 场景标签频次）
    fig02_horizon.png  误差随预测时域的曲线（0.1 s → 8 s）
    fig03_K.png        候选数 K 的影响（精度 / 多样性）
    fig04_steps.png    采样步数的影响（流匹配 vs 扩散）
    fig05_pareto.png   精度 vs 单次预测耗时
    fig06_tags.png     按场景标签分层的精度
    fig07_spread.png   候选铺开程度 + 分速度档的转弯覆盖
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "data"))
sys.path.insert(0, os.path.join(CODE, "models"))
sys.path.insert(0, os.path.join(CODE, "metrics"))

from dataset import MiniTrajDataset                # noqa: E402
from model import TrajectoryModel                  # noqa: E402
from train import cv_baseline                      # noqa: E402
from diag_turn import heading_change               # noqa: E402

CREAM, INK, GREY = "#F1EDE3", "#26231E", "#8A8378"
BLUE, ORANGE, RED, PURPLE, GREEN = ("#2E5E8C", "#E85D26", "#C0392B",
                                    "#8E44AD", "#3F7A47")
COL = {"reg": PURPLE, "fm": ORANGE, "ddpm": RED}
LAB = {"reg": "regression", "fm": "flow matching", "ddpm": "diffusion"}
STEPS = {"reg": 1, "fm": 4, "ddpm": 15}
OBJS = ("reg", "fm", "ddpm")
LAT_MS = {"reg": 6.2, "fm": 18.9, "ddpm": 59.5}      # batch-1, measured


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def style(ax, title, xl=None, yl=None, legend=True):
    ax.set_facecolor(CREAM)
    ax.grid(color="#CFC8B8", lw=0.6, alpha=0.7)
    ax.tick_params(labelsize=9, colors="#5C574E")
    for sp in ax.spines.values():
        sp.set_color("#CFC8B8")
    ax.set_title(title, fontsize=11, color=INK)
    if xl:
        ax.set_xlabel(xl, fontsize=9.5, color=INK)
    if yl:
        ax.set_ylabel(yl, fontsize=9.5, color=INK)
    if legend:
        ax.legend(fontsize=8.5, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=1536)
    ap.add_argument("--n-steps", type=int, default=384)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="_lp2")
    ap.add_argument("--out-dir", default="/home/parick/dsh_ml/figs")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(a.out_dir, exist_ok=True)
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    sc = ds.scale
    rng = np.random.default_rng(2024)
    idx = np.sort(rng.choice(len(ds), size=min(a.n, len(ds)), replace=False))
    items = [ds[int(i)] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    batch["tags"] = torch.stack([it["tags"] for it in items]).to(dev)
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    cv = cv_baseline(batch, ds.scale)[:, 0]
    print(f"split {a.split}  n={len(idx)}  K={a.K}  suffix='{a.suffix}'")

    # ---------- 1. 数据概览
    hist = batch["hist"].cpu().numpy() * sc
    v0 = np.linalg.norm(hist[:, -1, 2:4], axis=-1)
    tr = np.array([heading_change(t) for t in gt])
    dist = np.linalg.norm(gt[:, -1], axis=-1)
    tags = batch["tags"].cpu().numpy()
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8), facecolor=CREAM)
    ax[0][0].hist(v0, bins=40, color=BLUE, alpha=0.85)
    style(ax[0][0], "Speed at the prediction moment", "speed (m/s)", "scenes", False)
    ax[0][1].hist(tr, bins=50, color=ORANGE, alpha=0.85)
    style(ax[0][1], "Total heading change over the 8 s future",
          "heading change (deg)", "scenes", False)
    ax[1][0].hist(dist, bins=40, color=GREEN, alpha=0.85)
    style(ax[1][0], "How far the car actually travels in 8 s",
          "distance (m)", "scenes", False)
    tot = tags.sum(axis=0)
    top = np.argsort(tot)[::-1][:10]
    names = [ds.vocab[i][:26] for i in top]
    ax[1][1].barh(range(len(top))[::-1], tot[top], color=PURPLE, alpha=0.85)
    ax[1][1].set_yticks(range(len(top))[::-1])
    ax[1][1].set_yticklabels(names, fontsize=8)
    style(ax[1][1], "Most frequent scenario tags", "scenes", None, False)
    fig.suptitle(f"nuPlan mini, split {a.split} - what the data looks like "
                 f"(n={len(idx)})", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(a.out_dir, "fig01_data.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # ---------- 2/3/6/7 需要模型预测
    P = {}
    for o in OBJS:
        m = load(a.runs, o, dev, a.suffix)
        outs = []
        for s0 in range(0, len(items), 256):
            b = {k: v[s0:s0 + 256] for k, v in batch.items()}
            with torch.no_grad():
                pr, _ = m.sample(b, K=a.K if o != "reg" else 1, steps=STEPS[o])
            outs.append(ds.denorm_future(pr.cpu().numpy()))
        P[o] = np.concatenate(outs)
        del m
        print(f"  {o}: {P[o].shape}")

    # 2. 误差随时域
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=CREAM)
    ts = np.arange(1, 81) * 0.1
    for nm, pr, c in (("keep current speed", cv[:, None], GREEN),
                      *[(LAB[o], P[o], COL[o]) for o in OBJS]):
        # 逐点误差 (N,K,T)；用"整体平均误差最小"的那条候选来画曲线，
        # 这样曲线与该模型的 minADE 口径一致（不能逐点取 min，那会虚低）。
        pc = np.linalg.norm(pr - gt[:, None], axis=-1)          # (N,K,T)
        bi = pc.mean(-1).argmin(axis=1)                         # (N,)
        e = pc[np.arange(pc.shape[0]), bi]                      # (N,T)
        ax[0].plot(ts, e.mean(0), color=c, lw=2.2, label=nm)
    ax[0].fill_between(ts, 0, np.linalg.norm(gt, axis=-1).mean(0),
                       color="#CFC8B8", alpha=0.35, label="mean distance travelled")
    style(ax[0], "Error grows with the prediction horizon",
          "time ahead (s)", "position error (m)")
    for o in OBJS:
        pc = np.linalg.norm(P[o] - gt[:, None], axis=-1)
        bi = pc.mean(-1).argmin(axis=1)
        e = pc[np.arange(pc.shape[0]), bi]
        ax[1].plot(ts, e.mean(0), color=COL[o], lw=2.2, label=LAB[o])
    style(ax[1], "Same plot, zoomed on the three models",
          "time ahead (s)", "position error (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig02_horizon.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # 3. K 扫描（复用在 K=6 下采样的 6 条候选）
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=CREAM)
    ks = [1, 2, 3, 4, 5, 6]
    for o in ("fm", "ddpm"):
        ade, fde, div = [], [], []
        for k in ks:
            p = P[o][:, :k]
            ade.append(np.linalg.norm(p - gt[:, None], axis=-1).mean(-1).min(-1).mean())
            fde.append(np.linalg.norm(p[:, :, -1] - gt[:, None, -1],
                                      axis=-1).min(-1).mean())
            if k == 1:
                div.append(0.0)
            else:
                e = p[:, :, -1, :]
                d = np.linalg.norm(e[:, :, None] - e[:, None, :], axis=-1)
                div.append(d[:, np.triu_indices(k, 1)[0], np.triu_indices(k, 1)[1]].mean())
        ax[0].plot(ks, ade, "o-", color=COL[o], lw=2, label=f"{LAB[o]} minADE")
        ax[0].plot(ks, fde, "s--", color=COL[o], lw=2, alpha=0.6,
                   label=f"{LAB[o]} minFDE")
        ax[1].plot(ks, div, "o-", color=COL[o], lw=2, label=LAB[o])
    style(ax[0], "More candidates help, with diminishing returns",
          "number of candidates K", "error (m)")
    style(ax[1], "Diversity grows roughly linearly with K",
          "number of candidates K", "mean pairwise endpoint distance (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig03_K.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # 4. 步数扫描（小样本）
    sub = {k: v[:a.n_steps] for k, v in batch.items()}
    gts = ds.denorm_future(sub["future"].cpu().numpy())
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=CREAM)
    steps_list = [1, 2, 4, 8, 16, 32]
    for o in ("fm", "ddpm"):
        m = load(a.runs, o, dev, a.suffix)
        ys = []
        for st in steps_list:
            pr, _ = m.sample(sub, K=4, steps=st)
            p = ds.denorm_future(pr.cpu().numpy())
            ys.append(np.linalg.norm(p - gts[:, None], axis=-1).mean(-1).min(-1).mean())
        ax.plot(steps_list, ys, "o-", color=COL[o], lw=2.2, label=LAB[o])
        print(f"  steps sweep {o}: " + "  ".join(f"{s}:{y:.3f}"
                                                for s, y in zip(steps_list, ys)))
        del m
    ax.axhline(np.linalg.norm(cv[:a.n_steps, None] - gts[:, None],
                              axis=-1).mean(), color=GREEN, ls=":", lw=1.6,
               label="keep current speed")
    style(ax, "Sampling steps saturate quickly (K=4)",
          "number of refinement steps", "minADE (m)")
    ax.set_xscale("log", base=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig04_steps.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # 5. Pareto：精度 vs 耗时
    fig, ax = plt.subplots(figsize=(7.8, 4.9), facecolor=CREAM)
    base = [("keep current speed",
             np.linalg.norm(cv[:, None] - gt[:, None], axis=-1).mean(),
             1.0, GREEN)]
    for o in OBJS:
        ade = np.linalg.norm(P[o] - gt[:, None], axis=-1).mean(-1).min(-1).mean()
        base.append((LAB[o], ade, LAT_MS[o], COL[o]))
    for nm, ade, lat, c in base:
        ax.scatter([lat], [ade], s=150, color=c, zorder=5, edgecolor="white", lw=1.5)
        ax.annotate(f"{nm}\n{ade:.2f} m / {lat:.0f} ms", (lat, ade),
                    textcoords="offset points", xytext=(9, 6), fontsize=8.5, color=INK)
    ax.plot([b[2] for b in base], [b[1] for b in base], "--",
            color=GREY, lw=1.3, alpha=0.8, zorder=1)
    style(ax, "Accuracy versus cost per prediction", "time per prediction (ms, log)",
          "minADE@6 (m)", False)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 130)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig05_pareto.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # 6. 按场景标签分层
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=CREAM)
    keep = [(i, ds.vocab[i]) for i in top]
    w = 0.26
    for k, o in enumerate(OBJS):
        vals, cnts = [], []
        for ti, _nm in keep:
            msk = tags[:, ti] > 0
            cnts.append(int(msk.sum()))
            if msk.sum() < 15:
                vals.append(np.nan)
                continue
            vals.append(np.linalg.norm(P[o][msk] - gt[msk][:, None], axis=-1)
                        .mean(-1).min(-1).mean())
        xs = np.arange(len(keep)) + (k - 1) * w
        ax.bar(xs, vals, w, color=COL[o], alpha=0.88, label=LAB[o])
    ax.set_xticks(np.arange(len(keep)))
    ax.set_xticklabels([nm[:18] for _ti, nm in keep], rotation=32, ha="right", fontsize=8)
    style(ax, "Accuracy by scenario type (tags with n>=15 shown)",
          None, "minADE@6 (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig06_tags.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    # 7. 候选铺开程度 + 分档转弯覆盖
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), facecolor=CREAM)
    gy = gt[:, -1, 1]
    for o in ("fm", "ddpm"):
        e = P[o][:, :, -1, 1]
        span = e.max(1) - e.min(1)
        qs = np.percentile(np.abs(gy), np.linspace(5, 95, 10))
        mb, sb = [], []
        for q0, q1 in zip(qs[:-1], qs[1:]):
            msk = (np.abs(gy) >= q0) & (np.abs(gy) < q1)
            if msk.sum() > 15:
                mb.append(np.abs(gy)[msk].mean())
                sb.append(span[msk].mean())
        ax[0].plot(mb, sb, "o-", color=COL[o], lw=2, label=LAB[o])
    ax[0].plot(mb, mb, "s--", color=INK, lw=1.4, label="what would be needed")
    style(ax[0], "The 6 candidates are far too narrow sideways",
          "true sideways displacement (m)", "spread of the 6 candidates (m)")
    turn = np.abs(tr) > 30
    bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 100)]
    xs = np.arange(len(bins))
    for k, o in enumerate(("fm", "ddpm")):
        vals = []
        for lo, hi in bins:
            msk = (v0 >= lo) & (v0 < hi) & turn
            if msk.sum() < 25:
                vals.append(np.nan)
                continue
            p = P[o][msk]
            allc = np.array([[heading_change(t) for t in pk] for pk in p])
            ok = (np.sign(allc) == np.sign(tr[msk])[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr[msk])[:, None])
            vals.append(ok.any(axis=1).mean())
        ax[1].bar(xs + (k - 0.5) * 0.36, vals, 0.36, color=COL[o], alpha=0.88,
                  label=LAB[o])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], fontsize=8.5)
    style(ax[1], "Turn coverage by speed (fraction of real turns captured)",
          "speed at prediction time (m/s)", "coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig07_spread.png"), dpi=130, facecolor=CREAM)
    plt.close(fig)

    print(f"\n7 张图已写入 {a.out_dir}")
    for f in sorted(os.listdir(a.out_dir)):
        if f.endswith(".png"):
            print(f"  {f}  {os.path.getsize(os.path.join(a.out_dir, f))//1024} KB")


if __name__ == "__main__":
    main()
