"""图包后半部分：fig04–fig07。

与前半部分（fig01–fig03）分开，原因有两个：

1. **用户自己的 nuPlan 闭环仿真在占着 GPU**（`run_simulation.py`，6.6/8.2 GB），
   我的脚本和它抢显存导致一次 40 分钟的卡死。所以这里默认 `--device cpu`，
   彻底不碰 GPU，也不影响用户正在跑的仿真。
2. fig04（步数扫描）和 fig05（Pareto）**不需要重新推理** —— 这些数字之前已经量过
   并写在日志里，直接绘图即可，省掉整轮前向。

    python code/runners/make_figs_rest.py --only steps,pareto,tags,spread \
        --suffix _lp2 --n 192 --device cpu
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
LAT_MS = {"reg": 6.2, "fm": 18.9, "ddpm": 59.5}

# ---- 之前实测的数字（来源：训练/诊断日志，不是这里现算的）----
# fm/ddpm 的步数扫描：split 2, n=256, K=4（code/runners/diag_fm.py 的输出）
STEP_SWEEP = {
    "fm":   {1: 2.943, 2: 2.237, 4: 2.098, 8: 2.089, 16: 2.089, 32: 2.095},
    "ddpm": {1: 2.904, 2: 2.266, 4: 2.154, 8: 2.135, 16: 2.144, 32: 2.168},
}


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
    ap.add_argument("--report", default="/home/parick/dsh_ml/eval/report_split2_lp2.json")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="_lp2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="steps,pareto,tags,spread")
    ap.add_argument("--out-dir", default="/home/parick/dsh_ml/figs")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device(a.device)
    torch.manual_seed(0)
    os.makedirs(a.out_dir, exist_ok=True)
    want = set(x.strip() for x in a.only.split(",") if x.strip())
    print(f"device={dev}  suffix='{a.suffix}'  figures={sorted(want)}")

    # ---------------- fig04 步数扫描（用已测数字，不推理）
    if "steps" in want:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=CREAM)
        for o in ("fm", "ddpm"):
            xs = sorted(STEP_SWEEP[o])
            ys = [STEP_SWEEP[o][s] for s in xs]
            ax.plot(xs, ys, "o-", color=COL[o], lw=2.2, label=LAB[o])
            ax.annotate(f"{ys[2]:.2f}", (xs[2], ys[2]), textcoords="offset points",
                        xytext=(4, 8), fontsize=8.5, color=COL[o])
        ax.axhline(2.943, color=GREY, ls=":", lw=1.4,
                   label="1-step flow matching = regression")
        style(ax, "Sampling steps saturate: no gain beyond 4 (K=4, n=256)",
              "number of refinement steps", "minADE (m)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(STEP_SWEEP["fm"]))
        ax.set_xticklabels(sorted(STEP_SWEEP["fm"]))
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "fig04_steps.png"), dpi=130, facecolor=CREAM)
        plt.close(fig)
        print("  fig04_steps.png")

    # ---------------- fig05 Pareto（用评测报告里的数）
    if "pareto" in want:
        fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor=CREAM)
        pts = [("keep current speed", 5.937, 1.0, GREEN)]
        try:
            R = json.load(open(a.report, encoding="utf-8"))
            base = R["baseline"]["K=1"]
            pts[0] = ("keep current speed", base["minADE"], 1.0, GREEN)
            for o in OBJS:
                if o in R["models"]:
                    pts.append((LAB[o], R["models"][o]["agg"]["minADE"]["mean"],
                                R["models"][o]["latency_ms_b1"], COL[o]))
        except Exception as e:                                     # noqa: BLE001
            print(f"  ! 读不到 {a.report}（{e}），用内置数字")
            pts += [("regression", 2.823, LAT_MS["reg"], PURPLE),
                    ("flow matching", 1.826, LAT_MS["fm"], ORANGE),
                    ("diffusion", 1.736, LAT_MS["ddpm"], RED)]
        pts.sort(key=lambda x: x[2])
        ax.plot([p[2] for p in pts], [p[1] for p in pts], "--", color=GREY,
                lw=1.3, alpha=0.85, zorder=1)
        for nm, ade, lat, c in pts:
            ax.scatter([lat], [ade], s=170, color=c, zorder=5, edgecolor="white", lw=1.5)
            ax.annotate(f"{nm}\n{ade:.2f} m  /  {lat:.1f} ms", (lat, ade),
                        textcoords="offset points", xytext=(10, 4), fontsize=8.5,
                        color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 200)
        ax.set_ylim(1.2, 6.6)
        style(ax, "Accuracy versus cost  (K=6, held-out split)", 
              "time per prediction on one GPU (ms, log)", "minADE (m)", False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "fig05_pareto.png"), dpi=130, facecolor=CREAM)
        plt.close(fig)
        print("  fig05_pareto.png")

    # ---------------- fig06 / fig07 需要少量推理
    need_pred = ("tags" in want) or ("spread" in want)
    if need_pred:
        ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
        sc = ds.scale
        rng = np.random.default_rng(7)
        idx = np.sort(rng.choice(len(ds), size=min(a.n, len(ds)), replace=False))
        items = [ds[int(i)] for i in idx]
        batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
                 ("hist", "neigh", "neigh_mask", "future")}
        batch["tags"] = torch.stack([it["tags"] for it in items]).to(dev)
        gt = ds.denorm_future(batch["future"].cpu().numpy())
        hist = batch["hist"].cpu().numpy() * sc
        v0 = np.linalg.norm(hist[:, -1, 2:4], axis=-1)
        tr = np.array([heading_change(t) for t in gt])
        tags = batch["tags"].cpu().numpy()
        print(f"  推理 {len(idx)} 条（K={a.K}，device={dev}）")
        P = {}
        for o in OBJS:
            m = load(a.runs, o, dev, a.suffix)
            outs = []
            for s0 in range(0, len(items), 64):
                b = {k: v[s0:s0 + 64] for k, v in batch.items()}
                with torch.no_grad():
                    pr, _ = m.sample(b, K=a.K if o != "reg" else 1, steps=STEPS[o])
                outs.append(ds.denorm_future(pr.cpu().numpy()))
            P[o] = np.concatenate(outs)
            del m
            print(f"    {o} done")

        if "tags" in want:
            fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor=CREAM)
            tot = tags.sum(axis=0)
            top = np.argsort(tot)[::-1][:10]
            w = 0.26
            for k, o in enumerate(OBJS):
                vals = []
                for ti in top:
                    msk = tags[:, ti] > 0
                    if msk.sum() < 10:
                        vals.append(np.nan)
                    else:
                        vals.append(np.linalg.norm(P[o][msk] - gt[msk][:, None],
                                                   axis=-1).mean(-1).min(-1).mean())
                ax.bar(np.arange(len(top)) + (k - 1) * w, vals, w, color=COL[o],
                       alpha=0.88, label=LAB[o])
            ax.set_xticks(np.arange(len(top)))
            ax.set_xticklabels([ds.vocab[i][:18] for i in top], rotation=32,
                               ha="right", fontsize=8)
            style(ax, f"Accuracy by scenario type (tags with n>=10, n={len(idx)})",
                  None, "minADE@6 (m)")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out_dir, "fig06_tags.png"), dpi=130,
                        facecolor=CREAM)
            plt.close(fig)
            print("  fig06_tags.png")

        if "spread" in want:
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), facecolor=CREAM)
            gy = gt[:, -1, 1]
            mb = None
            for o in ("fm", "ddpm"):
                e = P[o][:, :, -1, 1]
                span = e.max(1) - e.min(1)
                qs = np.percentile(np.abs(gy), np.linspace(5, 95, 8))
                mb, sb = [], []
                for q0, q1 in zip(qs[:-1], qs[1:]):
                    msk = (np.abs(gy) >= q0) & (np.abs(gy) < q1)
                    if msk.sum() > 5:
                        mb.append(np.abs(gy)[msk].mean())
                        sb.append(span[msk].mean())
                ax[0].plot(mb, sb, "o-", color=COL[o], lw=2, label=LAB[o])
            if mb:
                ax[0].plot(mb, mb, "s--", color=INK, lw=1.4, label="what is needed")
            style(ax[0], "The 6 candidates are far too narrow sideways",
                  "true sideways displacement (m)", "spread of the 6 candidates (m)")
            turn = np.abs(tr) > 30
            bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 100)]
            xs = np.arange(len(bins))
            for k, o in enumerate(("fm", "ddpm")):
                vals = []
                for lo, hi in bins:
                    msk = (v0 >= lo) & (v0 < hi) & turn
                    if msk.sum() < 5:
                        vals.append(np.nan)
                        continue
                    p = P[o][msk]
                    allc = np.array([[heading_change(t) for t in pk] for pk in p])
                    ok = (np.sign(allc) == np.sign(tr[msk])[:, None]) & \
                         (np.abs(allc) >= 0.5 * np.abs(tr[msk])[:, None])
                    vals.append(ok.any(axis=1).mean())
                ax[1].bar(xs + (k - 0.5) * 0.36, vals, 0.36, color=COL[o],
                          alpha=0.88, label=LAB[o])
            ax[1].set_xticks(xs)
            ax[1].set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], fontsize=8.5)
            style(ax[1], "Turn coverage by speed (small n - indicative only)",
                  "speed at prediction time (m/s)", "coverage")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out_dir, "fig07_spread.png"), dpi=130,
                        facecolor=CREAM)
            plt.close(fig)
            print("  fig07_spread.png")

    print(f"\n完成，目录 {a.out_dir}:")
    for f in sorted(os.listdir(a.out_dir)):
        if f.endswith(".png"):
            print(f"  {f}  {os.path.getsize(os.path.join(a.out_dir, f))//1024} KB")


if __name__ == "__main__":
    main()
