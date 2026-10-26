"""逐场景数值解剖：为什么真值向下拐弯，而所有候选都向上拐弯？

不要靠看图猜。这个脚本把指定场景的**所有数值**打出来：
  · 目标车历史（米制）与历史末速度 → 匀速外推会往哪走
  · 真值未来（米制）→ 真实朝向变化
  · 历史末段方向 vs 未来首段方向（**检验 t0 处有没有不连续**）
  · 三个模型的 K 条候选
  · 邻居历史（看模型是不是在跟着邻居走）
并输出一张单场景放大图，把历史/真值/匀速/候选分开画。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_scene.py 10602
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

STEPS = {"reg": 1, "fm": 4, "ddpm": 15}


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="+", help="数据集下标（可多个）")
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="", help="checkpoint 目录后缀，如 _lp2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    sc = ds.scale
    models = {o: load(a.runs, o, dev, a.suffix) for o in ("reg", "fm", "ddpm")}

    for idx in a.idx:
        s = ds[int(idx)]
        hist = s["hist"].numpy() * sc
        fut = ds.denorm_future(s["future"].numpy())
        nb = s["neigh"].numpy() * sc
        nbm = s["neigh_mask"].numpy()
        log = ds.log_of(int(idx))
        batch = {k: v.unsqueeze(0).to(dev) for k, v in
                 [("hist", s["hist"]), ("neigh", s["neigh"]),
                  ("neigh_mask", s["neigh_mask"]), ("future", s["future"])]}

        print("=" * 78)
        print(f"下标 {idx}   log = {log}   邻居槽使用 {int(nbm.sum())}/10")
        print("=" * 78)

        # ---- 历史：位置与速度
        print("\n【目标车历史】(米制，t0 坐标系；末点=原点)")
        for k in (0, 5, 10, 15, 18, 19):
            print(f"  t={-(19-k)*0.1:+.1f}s  p=({hist[k,0]:+8.3f},{hist[k,1]:+8.3f})  "
                  f"v=({hist[k,2]:+7.3f},{hist[k,3]:+7.3f})  |v|={np.linalg.norm(hist[k,2:4]):6.3f}"
                  f"  朝向 {ang(hist[k,2:4]):+7.2f}°")
        v_last = hist[-1, 2:4].copy()
        # 用位置差分算一个独立的速度估计（不用缓存里的速度列）
        v_diff = (hist[-1, :2] - hist[-2, :2]) / 0.1
        print(f"  末速度（缓存）  |v|={np.linalg.norm(v_last):.3f} 朝向 {ang(v_last):+.2f}°")
        print(f"  末速度（差分）  |v|={np.linalg.norm(v_diff):.3f} 朝向 {ang(v_diff):+.2f}°")

        # ---- 真值未来
        print("\n【真值未来】(米制)")
        d0 = fut[0] - hist[-1, :2]                      # 未来第一步的位移
        print(f"  第 1 步位移 ({d0[0]:+.3f},{d0[1]:+.3f})  |Δ|={np.linalg.norm(d0):.3f} m "
              f"→ |v|={np.linalg.norm(d0)/0.1:.3f} m/s  朝向 {ang(d0):+.2f}°")
        for k in (0, 19, 39, 59, 79):
            print(f"  t=+{(k+1)*0.1:.1f}s  p=({fut[k,0]:+8.3f},{fut[k,1]:+8.3f})")
        print(f"  末点 ({fut[-1,0]:+.3f},{fut[-1,1]:+.3f})  |p|={np.linalg.norm(fut[-1]):.3f} m")

        # ---- 关键检验：历史末速度方向 vs 未来首步方向
        a_hist = ang(v_last)
        a_fut = ang(d0)
        da = (a_fut - a_hist + 180) % 360 - 180
        print(f"\n【t0 处连续性】历史末速度朝向 {a_hist:+.2f}°  vs  未来首步朝向 {a_fut:+.2f}°")
        print(f"  → 夹角 {da:+.2f}°   {'✓ 连续' if abs(da) < 30 else '⚠ 不连续！'}")
        # 一整段的朝向演化
        print("  真值未来的朝向演化：")
        for k in range(0, 80, 10):
            seg = (fut[min(k+10, 79)] - fut[k]) if k + 10 <= 79 else (fut[-1] - fut[-10])
            print(f"    t=+{(k+1)*0.1:.1f}..{(k+11)*0.1:.1f}s  朝向 {ang(seg):+8.2f}°")

        # ---- 匀速外推
        cv = np.zeros((80, 2))
        for k in range(80):
            cv[k] = hist[-1, :2] + v_last * (k + 1) * 0.1
        print(f"\n【匀速外推】末点 ({cv[-1,0]:+.2f},{cv[-1,1]:+.2f})  "
              f"minFDE {np.linalg.norm(cv[-1]-fut[-1]):.3f} m")

        # ---- 模型候选
        print("\n【模型候选】(K=%d)" % a.K)
        for o in ("reg", "fm", "ddpm"):
            K = 1 if o == "reg" else a.K
            with torch.no_grad():
                pr, nfe = models[o].sample(batch, K=K, steps=STEPS[o])
            p = ds.denorm_future(pr[0].cpu().numpy())
            ends = p[:, -1]
            ade = np.linalg.norm(p - fut[None], axis=-1).mean(-1)
            print(f"  {o:5s} (nfe {nfe:2d})  末点 y: " +
                  " ".join(f"{e[1]:+7.2f}" for e in ends))
            print(f"        {'':14s}末点 x: " +
                  " ".join(f"{e[0]:+7.2f}" for e in ends))
            print(f"        {'':14s}minADE {ade.min():7.3f}  minFDE "
                  f"{np.linalg.norm(p-fut[None],axis=-1)[:,-1].min():7.3f} m   "
                  f"首步朝向 " + " ".join(f"{ang(p[i,0]-hist[-1,:2]):+7.1f}°" for i in range(K)))
            del pr

        # ---- 邻居
        print("\n【邻居历史】(米制，末点位置与末速度)")
        for j in range(nb.shape[0]):
            if nbm[j]:
                pj = nb[j, -1, :2]
                vj = nb[j, -1, 2:4]
                print(f"  槽{j}: p=({pj[0]:+8.2f},{pj[1]:+8.2f})  "
                      f"v=({vj[0]:+7.2f},{vj[1]:+7.2f}) |v|={np.linalg.norm(vj):6.2f}  "
                      f"朝向 {ang(vj):+7.1f}°  |p|={np.linalg.norm(pj):6.1f} m")

        # ---- 单场景图
        if a.out:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="#F1EDE3")
            for ax, title in zip(axes, ("几何全貌", "真值 vs 匀速外推", "三个模型的候选")):
                ax.set_facecolor("#F1EDE3")
                ax.set_aspect("equal")
                ax.grid(color="#CFC8B8", lw=0.5, alpha=0.6)
                ax.set_title(title, fontsize=11, color="#26231E")
            for ax in axes:
                for j in range(nb.shape[0]):
                    if nbm[j]:
                        ax.plot(nb[j, :, 0], nb[j, :, 1], color="#B9B2A4", lw=1.0, zorder=1)
                ax.plot(hist[:, 0], hist[:, 1], color="#2E5E8C", lw=2.6, marker="o", ms=2.4,
                        zorder=5, label="history")
                ax.plot(fut[:, 0], fut[:, 1], color="#26231E", lw=2.8, zorder=6,
                        label="ground truth")
                ax.scatter([0], [0], s=40, color="#2E5E8C", zorder=7)
            axes[1].plot(cv[:, 0], cv[:, 1], color="#3F7A47", lw=2.0, ls="--",
                         label="constant velocity")
            for o, c in (("reg", "#8E44AD"), ("fm", "#E85D26"), ("ddpm", "#C0392B")):
                K = 1 if o == "reg" else a.K
                with torch.no_grad():
                    pr, _ = models[o].sample(batch, K=K, steps=STEPS[o])
                p = ds.denorm_future(pr[0].cpu().numpy())
                for k in range(p.shape[0]):
                    axes[2].plot(p[k, :, 0], p[k, :, 1], color=c, lw=1.4, alpha=0.75)
                axes[2].plot([], [], color=c, lw=1.6, label=o)
                del pr
            for ax in axes:
                ax.legend(fontsize=8, loc="best")
            fig.suptitle(f"idx {idx}  log {log[:34]}  GT end ({fut[-1,0]:.1f},{fut[-1,1]:.1f})  "
                         f"hist v dir {a_hist:+.1f}°  fut dir {a_fut:+.1f}°  Δ{da:+.1f}°",
                         fontsize=11, color="#26231E")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out = a.out if len(a.idx) == 1 else a.out.replace(".png", f"_{idx}.png")
            fig.savefig(out, dpi=140, facecolor="#F1EDE3")
            print(f"\n图已写入 {out}")


if __name__ == "__main__":
    main()
