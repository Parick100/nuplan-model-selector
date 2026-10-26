"""标定运动学指标的阈值。

**问题**（审查发现）：原实现取 A_MAX = 4 m/s²、J_MAX = 8 m/s³ 直接在离散差分上
判定，结果**真值自己就有 100% 的 jerk 超限率** —— 一个在真值上返回 1.000 的指标
没有任何信息量。原因是离散三阶差分把厘米级的标注噪声放大 1/dt³ = 1000 倍：

    |Δ³p| ~ 0.01 m  ⇒  |j| ~ 0.01 / 0.001 = 10 m/s³  >  J_MAX = 8

所以要么先低通去噪，要么把阈值按真值分布标定。本脚本同时测这两件事：
对不同的低通阶数 M，报告真值的 |a| / |j| 分位数与超限率，选出让指标有分辨力的设置。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/metrics/calibrate_kinematic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from dataset import MiniTrajDataset            # noqa: E402
from model import dct_lowpass                  # noqa: E402

ROOT = "/home/parick/dsh_ml/cache"
DT = 0.1
N = 20000


def derivs(p, dt=DT):
    v = np.diff(p, axis=-2) / dt
    a = np.diff(v, axis=-2) / dt
    j = np.diff(a, axis=-2) / dt
    return (np.linalg.norm(a, axis=-1).max(-1),
            np.linalg.norm(j, axis=-1).max(-1))


def lowpass(p, m):
    if m >= p.shape[-2]:
        return p
    P = dct_lowpass(p.shape[-2], m).numpy().astype(np.float64)
    return np.einsum("ij,bjc->bic", P, p)


def main():
    ds = MiniTrajDataset(ROOT, split=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N, len(ds)), replace=False)
    gt = np.stack([ds.denorm_future(ds[int(i)]["future"].numpy()) for i in idx])
    print(f"真值 {gt.shape}   逐点标注噪声量级可通过 |Δ²p| 观察")

    raw_a, raw_j = derivs(gt)
    print(f"\n{'M':>4} {'|a| p50':>9} {'p95':>8} {'max':>8} {'>4 m/s²':>9} "
          f"{'|j| p50':>9} {'p95':>8} {'max':>8} {'>8 m/s³':>9}")
    best = None
    for m in (0, 32, 40, 48, 56, 64, 72, 80):
        s = gt if m == 0 else lowpass(gt, m)
        a, j = derivs(s)
        row = (m, np.median(a), np.percentile(a, 95), a.max(),
               (a > 4.0).mean(), np.median(j), np.percentile(j, 95), j.max(),
               (j > 8.0).mean())
        print(f"{row[0]:>4} {row[1]:>9.3f} {row[2]:>8.3f} {row[3]:>8.2f} "
              f"{row[4]:>9.3f} {row[5]:>9.3f} {row[6]:>8.2f} {row[7]:>8.1f} "
              f"{row[8]:>9.3f}")
        # 选一个"真值超限率在 3–10%"的设置：指标有分辨力，又不至于全靠噪声
        if best is None or abs(row[8] - 0.05) < abs(best[8] - 0.05):
            best = row

    print(f"\n—— 未低通（M=0）的真值超限率: "
          f"加速度 {(raw_a > 4.0).mean():.3f}   jerk {(raw_j > 8.0).mean():.3f}")
    print("   若接近 1.000，说明原阈值在真值上就已经失效（指标无信息量）。")

    # 推荐：低通 M=56（保留 70% 模式，去掉白噪声但保留真实机动）后按真值分位标定
    s = lowpass(gt, 56)
    a, j = derivs(s)
    a_thr = float(np.percentile(a, 95))
    j_thr = float(np.percentile(j, 95))
    print(f"\n推荐设置：低通 M=56 + 阈值按真值 p95 标定")
    print(f"  A_MAX = {a_thr:.3f} m/s²   J_MAX = {j_thr:.3f} m/s³")
    print(f"  真值在该设置下的超限率: 加速度 {(a > a_thr).mean():.3f}  "
          f"jerk {(j > j_thr).mean():.3f}   （应 ≈0.05，这是相对指标的基准）")
    print(f"  真值 |a| p50/p95 {np.median(a):.3f}/{np.percentile(a,95):.3f} m/s²   "
          f"（物理合理：城市驾驶纵向加速度多在 1–3 m/s²）")


if __name__ == "__main__":
    main()
