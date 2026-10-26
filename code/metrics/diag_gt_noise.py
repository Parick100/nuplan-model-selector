"""诊断：真值未来轨迹的逐点噪声有多大？

`calibrate_kinematic.py` 显示真值自身的**峰值**加速度中位数是 22.7 m/s²、
p95 是 82.9 m/s² —— 物理上不可能（车辆纵向加速度上限约 10 m/s²）。
但同一批数据的**平均** |Δ²p| 只有 0.038 m（≈3.9 m/s²），说明不是整体偏移，
而是**几乎每条轨迹里都有零星的大跳变**。

本脚本要回答三个问题：
  Q1 跳变分布在时间轴的哪里（是否集中在首尾）？
  Q2 速度序列是否也被污染（|v| 出现不可能的值）？
  Q3 跳变是"单帧孤立"（标注量化/闪烁）还是"持续"（跟踪 ID 切换 / 帧序错乱）？

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/metrics/diag_gt_noise.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from dataset import MiniTrajDataset                       # noqa: E402

ROOT = "/home/parick/dsh_ml/cache"
DT = 0.1
N = 8000


def main():
    ds = MiniTrajDataset(ROOT, split=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N, len(ds)), replace=False)
    gt = np.stack([ds.denorm_future(ds[int(i)]["future"].numpy()) for i in idx])
    logs = [ds.log_of(int(i)) for i in idx]
    print(f"{gt.shape} 来自 {len(set(logs))} 个 log")

    # ---- 速度
    v = np.diff(gt, axis=-2) / DT                       # (N,79,2)
    sp = np.linalg.norm(v, axis=-1)                     # (N,79)
    print("\n=== Q2 速度 ===")
    print(f"  |v| p50 {np.median(sp):7.3f}  p95 {np.percentile(sp,95):7.3f}  "
          f"p99.9 {np.percentile(sp,99.9):7.3f}  max {sp.max():8.2f} m/s")
    print(f"  超过 40 m/s (144 km/h) 的点占比: {(sp>40).mean():.6f}")
    print(f"  超过 60 m/s (216 km/h) 的点占比: {(sp>60).mean():.6f}")
    print(f"  速度为 0 的点占比（可能表示帧被重复填充）: {(sp<1e-6).mean():.6f}")

    # ---- 二阶差分：定位跳变发生在时间轴何处
    acc = np.diff(v, axis=-2)                           # (N,78,2)
    am = np.linalg.norm(acc, axis=-1)                   # (N,78)
    frac_big = (am > 1.0).mean()                        # |Δ²p|>0.1 m 的比例
    print("\n=== Q1 大跳变的分布 ===")
    print(f"  单点 |Δ²p| > 0.1 m（即 |a| > 10 m/s²）的占比: {frac_big:.5f}")
    print(f"  单点 |Δ²p| > 1.0 m（即 |a| > 100 m/s²）的占比: {(am>1.0).mean():.6f}")
    print(f"  逐时间步的大跳变率（前 5 步 / 中段 / 后 5 步）:")
    print(f"    前5步 {(am[:,:5]>0.1).mean():.5f}   中段 {(am[:,30:50]>0.1).mean():.5f}"
          f"   后5步 {(am[:,-5:]>0.1).mean():.5f}")
    per_traj = (am > 0.1).sum(-1)
    print(f"  每条轨迹的大跳变次数: p50 {np.median(per_traj):.0f}  "
          f"p90 {np.percentile(per_traj,90):.0f}  max {per_traj.max()}")
    print(f"  完全没有大跳变的轨迹占比: {(per_traj==0).mean():.4f}")

    # ---- Q3：孤立单帧 vs 持续
    print("\n=== Q3 跳变的形状 ===")
    flat = am.reshape(-1)
    worst = np.argsort(flat)[-5:][::-1]
    for w in worst:
        n, k = divmod(int(w), am.shape[1])
        src = logs[n].split("/")[-1][:32]
        print(f"  log={src} traj#{n} step={k} |Δ²p|={flat[w]:.4f} m  "
              f"相邻步 |Δ²p| = {am[n, max(k-2,0):k+3].round(4).tolist()}")
        print(f"    p[{max(k-1,0)}..{k+3}] = {gt[n, max(k-1,0):k+4].round(4).tolist()}")

    # 孤立度：大跳变的邻居是否也大
    big = am > 0.1
    nb = np.zeros_like(big)
    nb[:, 1:] |= big[:, :-1]
    nb[:, :-1] |= big[:, 1:]
    both = big & nb
    print(f"\n  大跳变中「邻居也大」的比例（持续跳变）: {both.sum()/max(big.sum(),1):.4f}")
    print(f"  → 接近 0 表示都是**孤立单帧**（量化/闪烁），不是跟踪 ID 切换")

    # ---- 不连续性 vs 1e-3 网格
    print("\n=== 量化检查 ===")
    d = np.diff(gt, axis=-2)
    for q in (1e-6, 1e-4, 1e-3, 1e-2):
        print(f"  位移落在 {q:g} 网格上的比例: "
              f"{float((np.abs(d/q - np.round(d/q)) < 1e-6).mean()):.4f}")

    # ---- 与低通后对比：说明 DCT 约束同时在做去噪
    from model import dct_lowpass
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    for m in (24, 40, 56):
        P = dct_lowpass(80, m).numpy().astype(np.float64)
        s = np.einsum("ij,bjc->bic", P, gt)
        a2 = np.linalg.norm(np.diff(s, n=2, axis=-2), axis=-1)
        print(f"  低通 M={m}: |Δ²p| 均值 {a2.mean():.4f}  p95 {np.percentile(a2,95):.4f}  "
              f"max {a2.max():.4f} m   (>0.1m 占比 {(a2>0.1).mean():.5f})")


if __name__ == "__main__":
    main()
