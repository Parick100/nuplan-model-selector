"""分层场景画廊 + ASCII 校验 + 标注。

两个目的：
  1. **分层选场景**，而不是"按真值 FDE 分位挑 4 个"。审查早就指出 4 场景图会骗人：
     第一版那张图里恰好有一个真值右转的场景三个模型全错，看图会以为"模型全都朝一边拐"，
     而 1146 条大转弯样本的统计显示方向是对的。这里按"速度 × 转弯方向"分层，
     每层取该层**中位**样本，保证图有代表性且不挑好的。
  2. **ASCII BEV**：本会话的图像通道不可用，所以把同一批轨迹用字符网格打印出来，
     这样"预测到底往哪拐"可以在纯文本里核对，不需要看图下结论。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/viz_gallery.py \
        --suffix _lp2 --out /home/parick/dsh_ml/viz/gallery.png --ascii 3
"""
import argparse
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
from traj_metrics import min_ade, min_fde          # noqa: E402
from diag_turn import heading_change               # noqa: E402

CREAM, INK, GREY = "#F1EDE3", "#26231E", "#9A9384"
BLUE, ORANGE, PURPLE, RED, GREEN = "#2E5E8C", "#E85D26", "#8E44AD", "#C0392B", "#3F7A47"
STEPS = {"reg": 1, "fm": 4, "ddpm": 15}
COLORS = {"reg": PURPLE, "fm": ORANGE, "ddpm": RED}
OBJS = ["reg", "fm", "ddpm"]

# (名称, 速度下界, 速度上界, 转弯下界, 转完上界)  —— 转弯用绝对值区间
STRATA = [
    ("直行·中速",     3.0, 10.0, -15.0, 15.0),
    ("直行·高速",    10.0, 100.0, -15.0, 15.0),
    ("直行·低速",     0.3, 3.0, -15.0, 15.0),
    ("左转·中速",     3.0, 10.0, 30.0, 180.0),
    ("右转·中速",     3.0, 10.0, -180.0, -30.0),
    ("起步后大转弯",  0.0, 1.0, 45.0, 180.0),
    ("高速·大转弯",  10.0, 100.0, 25.0, 180.0),
]
# 图里用英文（matplotlib 默认字体没有中文字形，中文会渲染成方块）
EN_LABELS = ["straight / mid speed", "straight / high speed", "straight / low speed",
             "left turn / mid speed", "right turn / mid speed",
             "turn after standstill", "high speed / sharp turn"]


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def ascii_bev(hist, fut, preds, title, cols=52, rows=22):
    """字符网格：X=原点 h=历史 #=真值 r/f/d=各模型 minADE 最优候选。"""
    pts = [hist[:, :2], fut]
    for o in OBJS:
        if o in preds:
            pts.append(preds[o][0])          # 每个模型的最优候选
    allp = np.concatenate(pts, axis=0)
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    y0, y1 = allp[:, 1].min(), allp[:, 1].max()
    pad = 0.08 * max(x1 - x0, y1 - y0) + 1e-6
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    g = [[" "] * cols for _ in range(rows)]

    def put(p, ch, prio):
        cx = int((p[0] - x0) / (x1 - x0) * (cols - 1))
        cy = int((y1 - p[1]) / (y1 - y0) * (rows - 1))   # y 轴向上
        if 0 <= cx < cols and 0 <= cy < rows:
            cur = g[cy][cx]
            if cur == " " or prio >= PRIO.get(cur, 0):
                g[cy][cx] = ch

    PRIO = {" ": 0, "h": 1, "r": 2, "f": 3, "d": 4, "#": 5, "X": 6}
    for p in hist:
        put(p, "h", 1)
    for p in fut:
        put(p, "#", 5)
    for ch, o in (("r", "reg"), ("f", "fm"), ("d", "ddpm")):
        if o in preds:
            for p in preds[o][0]:
                put(p, ch, PRIO[ch])
    put(np.zeros(2), "X", 6)
    print(f"\n{title}   范围 x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] m")
    print("   " + "-" * cols)
    for row in g:
        print("   |" + "".join(row) + "|")
    print("   " + "-" * cols)
    print("   图例: X=原点(目标车 t0)  h=目标历史  #=真值未来  "
          "r/f/d = reg/流匹配/扩散 的 minADE 最优候选")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--probe", type=int, default=12000)
    ap.add_argument("--ascii", type=int, default=0, help="对前 N 个场景打印 ASCII BEV")
    ap.add_argument("--scenes", default="", help="直接指定下标（逗号分隔），覆盖分层")
    ap.add_argument("--per-stratum", type=int, default=1,
                    help="每个分层抽几个场景（按转弯幅度分位取）")
    ap.add_argument("--device", default="", help="cpu / cuda；留空自动选")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.device:
        dev = torch.device(a.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    sc = ds.scale
    models = {o: load(a.runs, o, dev, a.suffix) for o in OBJS}

    # ---- 选场景
    if a.scenes:
        idx = [int(x) for x in a.scenes.split(",")]
        labels = [f"手工指定 {i}" for i in idx]
        strata_of = [0] * len(idx)
    else:
        rng = np.random.default_rng(11)
        probe = rng.choice(len(ds), size=min(a.probe, len(ds)), replace=False)
        P = np.stack([ds.denorm_future(ds[int(i)]["future"].numpy()) for i in probe])
        H = np.stack([ds[int(i)]["hist"].numpy() * sc for i in probe])
        V = np.linalg.norm(H[:, -1, 2:4], axis=-1)
        TR = np.array([heading_change(t) for t in P])
        idx, labels, strata_of = [], [], []
        n_per = max(1, a.per_stratum)
        for si, (name, vlo, vhi, tlo, thi) in enumerate(STRATA):
            m = (V >= vlo) & (V < vhi) & (TR >= tlo) & (TR < thi)
            if m.sum() < 5:
                print(f"  ! 层「{name}」只有 {int(m.sum())} 条，跳过")
                continue
            sub = np.nonzero(m)[0]
            # 在该层内按转弯幅度取若干分位：每层抽 n_per 个，
            # 而不是只抽一个"中位"样本 —— 这样图里既有典型场景也有边角场景。
            absT = np.abs(TR[sub])
            qs = np.linspace(0.2, 0.8, n_per) if n_per > 1 else [0.5]
            picked = []
            for q in qs:
                tgt = np.quantile(absT, q)
                k = int(sub[np.argmin(np.abs(absT - tgt))])
                if k not in picked:
                    picked.append(k)
            for k in picked:
                idx.append(int(probe[k]))
                labels.append(f"{name} (该层 n={int(m.sum())}, |turn| {abs(TR[k]):.0f}°)")
                strata_of.append(si)
    print(f"\n选中 {len(idx)} 个场景：")
    for i, l in zip(idx, labels):
        print(f"  idx {i:>6}  {l}")

    # ---- 推理
    rows = []
    for i in idx:
        s = ds[int(i)]
        hist = s["hist"].numpy() * sc
        fut = ds.denorm_future(s["future"].numpy())
        nb = s["neigh"].numpy() * sc
        nbm = s["neigh_mask"].numpy()
        batch = {k: v.unsqueeze(0).to(dev) for k, v in
                 [("hist", s["hist"]), ("neigh", s["neigh"]),
                  ("neigh_mask", s["neigh_mask"]), ("future", s["future"])]}
        preds, stats = {}, {}
        for o in OBJS:
            K = 1 if o == "reg" else a.K
            with torch.no_grad():
                pr, nfe = models[o].sample(batch, K=K, steps=STEPS[o])
            p = ds.denorm_future(pr[0].cpu().numpy())
            ade = np.linalg.norm(p - fut[None], axis=-1).mean(-1)
            j = int(np.argmin(ade))
            preds[o] = p                                 # 全部 K 条（画图用）
            stats[o] = dict(ade=float(ade[j]),
                            fde=float(np.linalg.norm(p[j, -1] - fut[-1])),
                            all_ade=ade, nfe=nfe, n=K, best=j)
        rows.append(dict(idx=int(i), hist=hist, fut=fut, nb=nb, nbm=nbm,
                         preds=preds, stats=stats,
                         v0=float(np.linalg.norm(hist[-1, 2:4])),
                         turn=float(heading_change(fut))))
        print(f"\nidx {i:>6}  v(t0) {rows[-1]['v0']:6.2f} m/s  "
              f"真值转弯 {rows[-1]['turn']:+7.1f}°  真值末点 "
              f"({fut[-1,0]:+7.2f},{fut[-1,1]:+7.2f}) |p|={np.linalg.norm(fut[-1]):6.2f} m")
        for o in OBJS:
            st = stats[o]
            b = st["best"]
            print(f"    {o:5s} K={st['n']:2d} nfe {st['nfe']:3d}  "
                  f"minADE {st['ade']:7.3f}  minFDE {st['fde']:7.3f}  "
                  f"最优候选末点 ({preds[o][b,-1,0]:+7.2f},{preds[o][b,-1,1]:+7.2f})  "
                  f"该候选转弯 {heading_change(preds[o][b]):+7.1f}°")

    # ---- ASCII
    for r in rows[:a.ascii]:
        ascii_bev(r["hist"], r["fut"],
                  {o: r["preds"][o][[r["stats"][o]["best"]]] for o in OBJS},
                  f"idx {r['idx']}  v0={r['v0']:.2f} m/s  真值转弯={r['turn']:+.1f}°")

    # ---- PNG
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(14.5, 3.3 * n), facecolor=CREAM,
                                 squeeze=False)
        for r, row in enumerate(rows):
            for c in range(4):
                ax = axes[r][c]
                ax.set_facecolor(CREAM)
                for sp in ax.spines.values():
                    sp.set_color("#CFC8B8")
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(color="#CFC8B8", lw=0.5, alpha=0.55)
                ax.tick_params(labelsize=7, colors=GREY)
                for j in range(row["nb"].shape[0]):
                    if row["nbm"][j]:
                        ax.plot(row["nb"][j, :, 0], row["nb"][j, :, 1], color="#B9B2A4",
                                lw=1.0, alpha=0.8, zorder=1)
                ax.plot(row["hist"][:, 0], row["hist"][:, 1], color=BLUE, lw=2.4,
                        marker="o", ms=2.2, zorder=5)
                ax.plot(row["fut"][:, 0], row["fut"][:, 1], color=INK, lw=2.8, zorder=6)
                ax.scatter([0], [0], s=34, color=BLUE, zorder=7)
                if c == 0:
                    ax.set_title(f"#{r+1} ground truth   v0={row['v0']:.1f} m/s  "
                                 f"turn {row['turn']:+.0f} deg", fontsize=9, color=INK)
                else:
                    o = OBJS[c - 1]
                    st = row["stats"][o]
                    p = row["preds"][o]
                    for k in range(p.shape[0]):
                        isbest = (k == st["best"])
                        ax.plot(p[k, :, 0], p[k, :, 1], color=COLORS[o],
                                lw=2.6 if isbest else 1.1,
                                alpha=1.0 if isbest else 0.55,
                                zorder=4 if isbest else 3)
                    ax.set_title(f"{o}   minADE {st['ade']:.2f}   minFDE {st['fde']:.2f}"
                                 f"   (K={st['n']}, NFE {st['nfe']})",
                                 fontsize=9, color=INK)
            # 左侧标注（英文，避免 matplotlib 缺中文字形渲染成方块）
            axes[r][0].set_ylabel(EN_LABELS[strata_of[r] % len(EN_LABELS)],
                                  fontsize=9, color=INK)
        fig.suptitle("nuPlan mini - split 2 (unseen by all three models) - target-agent frame at t0\n"
                     "black = ground truth, blue = history, grey = neighbours, "
                     "thin coloured = all K candidates, thick = minADE-best candidate",
                     fontsize=10.5, color=INK, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        fig.savefig(a.out, dpi=130, facecolor=CREAM)
        print(f"\n图已写入 {a.out}  ({os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
