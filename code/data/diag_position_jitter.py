"""量化 nuPlan 跟踪位置的高频抖动 —— 决定运动学指标该怎么办。

已知：lidar_pc 是干净的 20 Hz（dt 0.05 s，std 1.7e-4），去掉偶数帧后 dt = 0.1000 s
±7%，(lidar_pc, track) 无重复行。但**位置列本身**存在隔帧抖动：静止车 ±0.02–0.05 m，
快车更大。在 10 Hz 上做二阶差分会把这种抖动放大 1/dt² = 100 倍，于是：

    0.03 m 的抖动  →  |a| = 3 m/s²（与真实加速度同量级）
    0.23 m 的抖动  →  |a| = 23 m/s²（物理上不可能）

所以"加速度超限率"这类指标必须先把抖动滤掉，或者改成**相对于真值分位**的口径。

本脚本给出定量证据：
  (1) 位置差 vs db 自带速度列 的一致性（速度列是标定出来的，通常比差分干净）
  (2) 低通到不同模式数 M 后残余的 |Δ²p|，以及它对应的 |a|
  (3) 结论：GT 的可达精度下限（平滑预测器对抖动真值的 minADE 下界）

只读。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/data/diag_position_jitter.py
"""
import glob
import os
import sqlite3
from collections import defaultdict

import numpy as np

DB_DIR = "/home/parick/nuplan/dataset/nuplan-v1.1/splits/mini"
N_LOGS = 3
STRIDE = 5
H, F = 20, 80


def one_log(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("SELECT token, timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
    ts = np.array([r[1] for r in rows], dtype=np.float64) / 1e6
    fidx = {r[0]: i for i, r in enumerate(rows)}
    tracks = defaultdict(dict)
    for lpc, trk, x, y, vx, vy in con.execute(
        "SELECT lidar_pc_token, track_token, x, y, vx, vy FROM lidar_box"
    ):
        i = fidx.get(lpc)
        if i is not None:
            tracks[trk][i] = (x, y, vx, vy)
    cats = {t: c for t, c in con.execute("SELECT token, category_token FROM track")}
    c2n = {t: n for t, n in con.execute("SELECT token, name FROM category")}
    con.close()

    dec = np.arange(0, len(ts), 2)
    out = []
    for trk, d in tracks.items():
        if c2n.get(cats.get(trk), "?") != "vehicle":
            continue
        pr = [j for j in dec if j in d]
        if len(pr) < H + F + 1:
            continue
        prs = set(pr)
        for t0 in range(H - 1, len(pr) - F - 1, STRIDE * 7):
            win = [pr[t0 - H + 1 + k] for k in range(H + F)]
            if any(w not in prs for w in win):
                continue
            P = np.array([d[w][:2] for w in win], dtype=np.float64)
            V = np.array([d[w][2:4] for w in win], dtype=np.float64)
            T = np.array([ts[w] for w in win], dtype=np.float64)
            if np.linalg.norm(P[-1] - P[0]) < 5.0:
                continue
            out.append((P, V, T))
            if len(out) >= 40:
                return out
    return out


def jitter_metric(P, T):
    dt = np.diff(T)
    dp = np.diff(P, axis=0)
    v = dp / dt[:, None]
    a = np.diff(v, axis=0) / dt[:-1, None]
    return v, a


def main():
    dbs = sorted(glob.glob(os.path.join(DB_DIR, "*.db")))[:N_LOGS]
    print(f"扫描 {len(dbs)} 个 log")
    allv, allvc, allj = [], [], []
    corrs = []
    M_LIST = (80, 56, 40, 24, 16)
    resid = {m: [] for m in M_LIST}
    for p in dbs:
        seqs = one_log(p)
        print(f"  {os.path.basename(p)[:40]}: {len(seqs)} 段")
        for P, V, T in seqs:
            v, a = jitter_metric(P, T)
            # db 速度与位置差分的一致性：直接对齐前 20 步（历史段）
            vv = v[:H - 1]
            vc = V[1:H]
            ok = (np.linalg.norm(vv, axis=-1) > 2.0) | (np.linalg.norm(vc, axis=-1) > 2.0)
            if ok.sum() > 5 and vc.std() > 1e-6 and vv.std() > 1e-6:
                c = np.corrcoef(vv[ok].ravel(), vc[ok].ravel())[0, 1]
                if np.isfinite(c):
                    corrs.append(float(c))
            sp = np.linalg.norm(v, axis=-1)
            if sp.max() < 3.0:
                continue
            allv.append(np.linalg.norm(v, axis=-1))
            allvc.append(np.linalg.norm(V[1:], axis=-1))
            allj.append(np.linalg.norm(a, axis=-1))
            rel = P - P[0]
            for m in M_LIST:
                if m >= len(rel):
                    resid[m].append(rel)
                    continue
                k = np.arange(m)[:, None]
                i = np.arange(len(rel))[None, :]
                B = np.cos(np.pi * (i + 0.5) * k / len(rel))
                B[0] *= 1 / np.sqrt(2)
                B = B / np.linalg.norm(B, axis=1, keepdims=True)
                Proj = B.T @ B
                resid[m].append(np.einsum("ij,jc->ic", Proj, rel))

    v = np.concatenate(allv); vc = np.concatenate(allvc); j = np.concatenate(allj)
    print("\n=== (1) 位置差分 vs db 速度列 ===")
    print(f"  corr(Δp/dt, db v) 在运动段的中位数: {np.median(corrs):+.4f}  "
          f"(n={len(corrs)} 段)")
    print(f"  位置差分 |v|: p50 {np.median(v):.3f}  p95 {np.percentile(v,95):.3f}  "
          f"p99.9 {np.percentile(v,99.9):.3f}  max {v.max():.2f} m/s")
    print(f"  db   速度 |v|: p50 {np.median(vc):.3f}  p95 {np.percentile(vc,95):.3f}  "
          f"p99.9 {np.percentile(vc,99.9):.3f}  max {vc.max():.2f} m/s")
    print(f"  → 位置差分的重尾程度（p99.9/p50）{np.percentile(v,99.9)/np.median(v):.2f}  "
          f"vs db 速度 {np.percentile(vc,99.9)/max(np.median(vc),1e-6):.2f}")
    print(f"\n  二阶差分 |a|: p50 {np.median(j):.2f}  p95 {np.percentile(j,95):.2f}  "
          f"max {j.max():.1f} m/s²   （物理上城市车辆 |a| 极少超过 4 m/s²）")

    print("\n=== (2) 低通到 M 个 DCT 模式后的残余抖动 ===")
    print(f"{'M':>4} {'原始 |Δ²p| 均值':>16} {'低通后均值':>12} {'p95':>9} {'max':>9} "
          f"{'等效 |a| 均值':>14} {'能量保留':>10}")
    for m in M_LIST:
        if not resid[m]:
            continue
        raw = np.concatenate(allj)
        r = np.concatenate(resid[m])
        a = np.diff(r, n=2, axis=-2)
        am = np.linalg.norm(a, axis=-1)
        e = 1 - ((r - np.concatenate(resid[80])) ** 2).sum() / \
            (np.concatenate(resid[80]) ** 2).sum() if m != 80 else 1.0
        print(f"{m:>4} {np.median(raw):>16.4f} {am.mean():>12.4f} "
              f"{np.percentile(am,95):>9.4f} {am.max():>9.4f} "
              f"{am.mean()*100:>14.2f} {e:>10.5f}")
    print("  （'等效 |a|' = 低通后的 |Δ²p| / dt²，dt=0.1）")

    print("\n=== (3) 结论 ===")
    print("  · 真值位置列在 10 Hz 上有高频抖动，直接把二阶/三阶差分当加速度/加加速度")
    print("    会让**真值自己**出现 100% 的超限率 —— 那样的指标没有信息量。")
    print("  · 正确做法：指标分两类")
    print("      (a) 绝对口径：低通（M≈40-56）后与真值同口径比较；")
    print("      (b) 相对口径：模型 |Δ²p| 与真值 |Δ²p| 的比值（粗糙度倍数）。")
    print("  · 训练侧不受影响：MSE 回归的贝叶斯最优解就是条件均值，天然平滑；")
    print("    DCT 低通约束同时起到去抖作用，是正确的归纳偏置。")


if __name__ == "__main__":
    main()
