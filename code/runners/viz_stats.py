"""统计图：单场景图会骗人，所以再给一张总体图。

四格：
 (a) 预测末点 y vs 真值末点 y 的二维密度 + 对角线 → 有没有系统性左右偏置
 (b) 候选横向跨度 vs 真值 |y_T| → "候选铺得够不够开"（这是诊断出的核心缺陷）
 (c) 按 t0 速度分档的转弯覆盖率（大转弯样本里至少一条候选同向且幅度 >=50%）
 (d) minADE vs NFE 的 Pareto 曲线（含匀速外推基线）→ 算力-精度取舍

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/viz_stats.py \
        --suffix _lp2 --out /home/parick/dsh_ml/viz/stats.png
"""
import argparse
import os
import sys
import json

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

STEPS = {"reg": 1, "fm": 4, "ddpm": 15}
COL = {"reg": "#8E44AD", "fm": "#E85D26", "ddpm": "#C0392B"}
BINS = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 100)]


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    sc = ds.scale
    rng = np.random.default_rng(5)
    idx = np.sort(rng.choice(len(ds), size=a.n, replace=False))
    items = [ds[int(i)] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    tr = np.array([heading_change(t) for t in gt])
    turn = np.abs(tr) > 30.0
    v0 = np.linalg.norm(batch["hist"].cpu().numpy()[:, -1, 2:4], axis=-1) * sc

    preds, ades = {}, {}
    for o in ("reg", "fm", "ddpm"):
        K = 1 if o == "reg" else a.K
        m = load(a.runs, o, dev, a.suffix)
        outs = []
        for s0 in range(0, len(items), 256):
            b = {k: v[s0:s0 + 256] for k, v in batch.items()}
            with torch.no_grad():
                pr, _ = m.sample(b, K=K, steps=STEPS[o])
            outs.append(ds.denorm_future(pr.cpu().numpy()))
        preds[o] = np.concatenate(outs)
        ades[o] = np.linalg.norm(preds[o] - gt[:, None], axis=-1).mean(-1).min(-1)
        del m
    cv = cv_baseline(batch, ds.scale)
    ade_cv = np.linalg.norm(cv - gt[:, None], axis=-1).mean(-1).min(-1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5), facecolor="#F1EDE3")
    for row in ax:
        for q in row:
            q.set_facecolor("#F1EDE3")
            q.grid(color="#CFC8B8", lw=0.5, alpha=0.6)
            q.tick_params(labelsize=9, colors="#6B655A")

    # ---- (a) 末点 y 散点
    A = ax[0][0]
    gy = gt[:, -1, 1]
    for o in ("fm", "ddpm"):
        by = preds[o][np.arange(len(gy)), np.argmin(
            np.linalg.norm(preds[o] - gt[:, None], axis=-1).mean(-1), axis=1), -1, 1]
        A.scatter(gy, by, s=4, alpha=0.35, color=COL[o], label=o)
    lim = [min(gy.min(), -20), max(gy.max(), 20)]
    A.plot(lim, lim, color="#26231E", lw=1.2, ls="--", label="identity")
    A.set_xlim(lim); A.set_ylim(lim)
    A.set_xlabel("ground-truth endpoint lateral y (m)")
    A.set_ylabel("predicted endpoint lateral y (m)")
    A.set_title("(a) endpoint lateral: no systematic left/right bias", fontsize=10)
    A.legend(fontsize=8)

    # ---- (b) 候选跨度 vs 真值横向量级
    B = ax[0][1]
    for o in ("fm", "ddpm"):
        span = preds[o][:, :, -1, 1].max(1) - preds[o][:, :, -1, 1].min(1)
        eb = np.percentile(np.abs(gy), np.linspace(5, 95, 10))
        mb, sb = [], []
        for q0, q1 in zip(eb[:-1], eb[1:]):
            mm = (np.abs(gy) >= q0) & (np.abs(gy) < q1)
            if mm.sum() > 20:
                mb.append(np.abs(gy)[mm].mean()); sb.append(span[mm].mean())
        B.plot(mb, sb, "o-", color=COL[o], label=f"{o} candidate y-span")
    B.plot(mb, mb, "s--", color="#26231E", label="ideal (span = |y_T|)")
    B.set_xlabel("ground-truth |lateral displacement| (m)")
    B.set_ylabel("mean candidate y-span (m)")
    B.set_title("(b) candidates are far too narrow laterally", fontsize=10)
    B.legend(fontsize=8)

    # ---- (c) 分档转弯覆盖
    C = ax[1][0]
    labels, series = [], {}
    for o in ("fm", "ddpm"):
        vals = []
        for lo, hi in BINS:
            m = (v0 >= lo) & (v0 < hi)
            tm = m & turn
            if tm.sum() < 30:
                vals.append(np.nan); continue
            p = preds[o][tm]
            allc = np.array([[heading_change(t) for t in pk] for pk in p])
            ok = (np.sign(allc) == np.sign(tr[tm])[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr[tm])[:, None])
            vals.append(ok.any(axis=1).mean())
        series[o] = vals
    ns = [int(((v0 >= lo) & (v0 < hi) & turn).sum()) for lo, hi in BINS]
    labels = [f"{lo}-{hi}\nn={n}" for (lo, hi), n in zip(BINS, ns)]
    x = np.arange(len(BINS)); w = 0.36
    for k, o in enumerate(("fm", "ddpm")):
        C.bar(x + (k - 0.5) * w, series[o], w, color=COL[o], alpha=0.85, label=o)
    C.set_xticks(x); C.set_xticklabels(labels, fontsize=8)
    C.set_ylabel("turn coverage")
    C.set_xlabel("|v(t0)| bin (m/s)  and  n of >30-deg-turn samples")
    C.set_title("(c) turn coverage by speed: worst at 1-3 m/s and >=10 m/s", fontsize=10)
    C.legend(fontsize=8)

    # ---- (d) Pareto
    D = ax[1][1]
    D.scatter([1], [ade_cv.mean()], s=110, marker="^", color="#3F7A47", zorder=5,
              label=f"constant velocity (ADE {ade_cv.mean():.2f})")
    rep = os.path.join("/home/parick/dsh_ml/eval", "report_split2_lp2.json")
    if os.path.exists(rep):
        R = json.load(open(rep, encoding="utf-8"))
        for o in ("reg", "fm", "ddpm"):
            if o in R["models"]:
                D.scatter([R["models"][o]["nfe"]], [R["models"][o]["agg"]["minADE"]["mean"]],
                          s=110, color=COL[o], zorder=5,
                          label=f"{o} (NFE {R['models'][o]['nfe']}, "
                                f"ADE {R['models'][o]['agg']['minADE']['mean']:.2f})")
    D.set_xscale("log")
    D.set_xlabel("NFE (network evaluations per scene, K included)")
    D.set_ylabel("minADE@K (m)")
    D.set_title("(d) accuracy vs compute: diffusion buys only ~5% over flow matching",
                fontsize=10)
    D.legend(fontsize=8)

    fig.suptitle(f"nuPlan mini - split {a.split} (unseen) - n={len(idx)} scenes, K={a.K} - "
                 f"model suffix '{a.suffix}'    (companion to the per-scene gallery)",
                 fontsize=11, color="#26231E")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        fig.savefig(a.out, dpi=130, facecolor="#F1EDE3")
        print(f"图已写入 {a.out} ({os.path.getsize(a.out)//1024} KB)")

    # 文字版结论（便于不看图也能核对）
    print(f"\n=== 文字核对 ===")
    for o in ("fm", "ddpm"):
        by = preds[o][np.arange(len(gy)), np.argmin(
            np.linalg.norm(preds[o] - gt[:, None], axis=-1).mean(-1), axis=1), -1, 1]
        print(f"  {o:5s} mean(pred_y − gt_y) = {float((by-gy).mean()):+7.3f} m   "
              f"corr = {float(np.corrcoef(by, gy)[0,1]):+.4f}   "
              f"候选 y 跨度中位 {float(np.median(preds[o][:,:,-1,1].max(1)-preds[o][:,:,-1,1].min(1))):6.2f} m")
    print(f"  真值 |y_T|: p50 {np.median(np.abs(gy)):.2f}  p95 {np.percentile(np.abs(gy),95):.2f} m")
    print(f"  转弯覆盖: " + "  ".join(
        f"{lo}-{hi}: fm {series['fm'][k]:.3f} ddpm {series['ddpm'][k]:.3f} (n={ns[k]})"
        for k, (lo, hi) in enumerate(BINS)))


if __name__ == "__main__":
    main()
