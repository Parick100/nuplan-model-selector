"""在真实 nuPlan 场景上可视化三个模型的预测。

对**同一个场景**并排画出：
    真值未来（黑）· K 条候选（橙）· 目标车历史（蓝）· 邻居历史（灰）
目标坐标系：原点 = 目标车 t0 位置，+x = 车头方向。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/visualize.py \
        --runs /home/parick/dsh_ml/runs --out /home/parick/dsh_ml/viz/pred.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "data"))
sys.path.insert(0, os.path.join(CODE, "models"))
sys.path.insert(0, os.path.join(CODE, "metrics"))

from dataset import MiniTrajDataset      # noqa: E402
from model import TrajectoryModel        # noqa: E402
from traj_metrics import min_ade, min_fde  # noqa: E402

CREAM, INK, GREY, BLUE, ORANGE, GREEN = ("#F1EDE3", "#26231E", "#9A9384",
                                        "#2E5E8C", "#E85D26", "#3F7A47")
OBJS = ["reg", "fm", "ddpm"]
TITLES = {"reg": "regression (1x)", "fm": "flow matching (4x)", "ddpm": "diffusion (15x)"}
STEPS = {"reg": 1, "fm": 4, "ddpm": 15}


def load(root, obj, dev, suffix=""):
    """用 checkpoint 里保存的 arch 构造模型；架构不符时跳过而不是崩溃。"""
    p = os.path.join(root, obj + suffix, "model.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=dev)
    arch = ck.get("arch")
    if arch is None:
        print(f"  ! {obj}: checkpoint 无 arch 字段（旧版），跳过")
        return None
    m = TrajectoryModel(**arch).to(dev)
    try:
        m.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        print(f"  ! {obj}: 架构与 checkpoint 不符，跳过 ({str(e).splitlines()[0][:70]})")
        return None
    m.eval()
    # 归一化 buffer 必须随 checkpoint 一起回来，否则输出会静默错位 —— 显式确认
    if not bool(torch.any(m.sd != 1.0)):
        print(f"  ! {obj}: mu/sd buffer 未从 checkpoint 恢复（归一化退回恒等），跳过")
        return None
    print(f"  {obj}: ok  code_hash={ck.get('code_hash','?')}  "
          f"steps={ck.get('args',{}).get('steps','?')}  basis_m={arch.get('basis_m','?')}  "
          f"dof={m.output_dim}")
    return m


def pick_scenes(ds, n, seed=0, spread=True):
    """挑 n 个场景。默认按**米制**真值 FDE 分位挑，覆盖容易/中等/困难。

    注意必须在米制下排序：归一化是逐时间步除以 sd，末点的 sd 与中间步不同，
    直接对归一化值排序会挑出与"实际难度"不一致的场景。
    """
    idx = np.arange(len(ds))
    rng = np.random.default_rng(seed)
    probe = rng.choice(idx, size=min(4000, len(idx)), replace=False)
    fdes = np.array([np.linalg.norm(ds.denorm_future(ds[i]["future"].numpy())[-1])
                     for i in probe])
    if not spread:
        return probe[:n]
    qs = np.linspace(0.35, 0.97, n)
    out = []
    for q in qs:
        tgt = np.quantile(fdes, q)
        out.append(int(probe[np.argmin(np.abs(fdes - tgt))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2, help="默认用选择器测试集（模型没见过）")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="", help="checkpoint 目录后缀，如 _lp2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    models = {o: load(a.runs, o, dev, a.suffix) for o in OBJS}
    have = [o for o in OBJS if models[o] is not None]
    if not have:
        sys.exit(f"no checkpoints under {a.runs}")
    print("loaded:", {o: (models[o].param_report()["total"] if models[o] else None) for o in OBJS})

    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=1)
    sc = ds.scale                      # 仅用于 hist / neigh（编码器输入）
    print(f"hist/neigh 全局标量 = {sc:.4f} m   逐元素归一化 = {ds.mu is not None}")
    idx = pick_scenes(ds, a.n, seed=7)
    print("场景下标:", idx)

    ncol = 1 + len(have)
    fig, axes = plt.subplots(a.n, ncol, figsize=(3.5 * ncol, 3.6 * a.n),
                             facecolor=CREAM, squeeze=False)
    rows = []
    # 平滑度：二阶差分向量的模长（≈加速度），越小越平滑
    smooth = {o: [] for o in have}
    gt_sm = []

    def wiggle(t):                                  # t: (..., T, 2)
        return np.linalg.norm(np.diff(t, n=2, axis=-2), axis=-1).mean()

    def cv_curve(hist_m, F=80, dt=0.1):
        """匀速外推参考线（米制）：末速度为 0 时退化为一个点，这种情况下不画。"""
        v = hist_m[-1, 2:4]
        if np.linalg.norm(v) < 0.3:
            return None
        return v[None, :] * (np.arange(1, F + 1)[:, None] * dt)

    for r, i in enumerate(idx):
        s = ds[i]
        hist = s["hist"].numpy() * sc                       # 米制
        fut = ds.denorm_future(s["future"].numpy())         # 米制
        nb = s["neigh"].numpy() * sc
        cv = cv_curve(hist)
        gt_sm.append(wiggle(fut))
        nbm = s["neigh_mask"].numpy()
        batch = {k: v.unsqueeze(0).to(dev) for k, v in
                 [("hist", s["hist"]), ("neigh", s["neigh"]),
                  ("neigh_mask", s["neigh_mask"]), ("future", s["future"])]}

        for c in range(ncol):
            ax = axes[r][c]
            ax.set_facecolor(CREAM)
            for sp in ax.spines.values():
                sp.set_color("#CFC8B8")
            # 邻居历史
            for j in range(nb.shape[0]):
                if nbm[j]:
                    ax.plot(nb[j, :, 0], nb[j, :, 1], color=GREY, lw=1.0, alpha=0.75, zorder=2)
            # 目标车历史
            ax.plot(hist[:, 0], hist[:, 1], color=BLUE, lw=2.2, zorder=4,
                    marker="o", ms=2.2)
            ax.plot(fut[:, 0], fut[:, 1], color=INK, lw=2.6, zorder=5, label="ground truth")
            if cv is not None:
                ax.plot(cv[:, 0], cv[:, 1], color=GREEN, lw=1.4, ls="--", alpha=0.9,
                        zorder=3, label="constant velocity")
            ax.scatter([0], [0], s=26, color=BLUE, zorder=6)

            if c == 0:
                ax.set_title("ground truth", fontsize=10, color=INK)
                rec = "—"
            else:
                o = have[c - 1]
                m = models[o]
                with torch.no_grad():
                    preds, nfe = m.sample(batch, K=a.K, steps=STEPS[o])
                p = ds.denorm_future(preds[0].cpu().numpy())     # 米制
                smooth[o].append(wiggle(p))
                for k in range(p.shape[0]):
                    ax.plot(p[k, :, 0], p[k, :, 1], color=ORANGE, lw=1.5, alpha=0.7, zorder=3)
                ade = float(min_ade(p[None], fut[None]).mean())
                fde = float(min_fde(p[None], fut[None]).mean())
                rec = f"ADE {ade:.2f}  FDE {fde:.2f}"
                ax.set_title(f"{TITLES[o]}   |   {rec}", fontsize=10, color=INK)

            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(color="#CFC8B8", lw=0.5, alpha=0.5)
            ax.tick_params(labelsize=7, colors=GREY)
        rows.append((int(i), float(np.linalg.norm(fut[-1]))))

    fig.suptitle(f"nuPlan mini · split {a.split} (unseen) · target-agent frame · "
                 f"K={a.K} candidates in orange · neighbours grey · "
                 f"constant velocity dashed",
                 fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(a.out, dpi=150, facecolor=CREAM)
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")
    print("场景 (下标, 真值 FDE m):", [(i, round(f, 1)) for i, f in rows])
    g0 = float(np.mean(gt_sm))
    print(f"\n平滑度 |d²p| （每步二阶差分模长，越小越平滑）：")
    print(f"  ground truth : {g0:.4f} m")
    for o in have:
        v = float(np.mean(smooth[o]))
        print(f"  {o:5s}        : {v:.4f} m   ({v/max(g0,1e-9):.1f}x GT)")


if __name__ == "__main__":
    main()
