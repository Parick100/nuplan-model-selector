"""直查 .db 原始表：真值未来为什么会有 ±1 m 的隔帧振荡？

假设：
  H-A `lidar_pc` 的 timestamp 不是均匀 20 Hz（有重复 / 缺口），
      于是 `dec = arange(0, n20, 2)` 取到的并不是等间隔 0.1 s。
  H-B `lidar_box` 里 (lidar_pc_token, track_token) 有重复行，
      于是 `tracks[trk][i]` 被后来的行覆盖，且顺序与时间不对应。
  H-C `lidar_box.x/y` 本身就是隔帧交替的两个估计（nuPlan 存的是两种来源的框）。

只读，不写任何 WSL 文件。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/data/diag_raw_db.py 2021.06.14.16.48.02_veh-12
"""
import glob
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

DB_DIR = "/home/parick/nuplan/dataset/nuplan-v1.1/splits/mini"


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "2021.06.14.16.48.02_veh-12"
    hits = [p for p in glob.glob(os.path.join(DB_DIR, "*.db")) if pat in os.path.basename(p)]
    if not hits:
        hits = [p for p in glob.glob(os.path.join(DB_DIR, "**", "*.db"), recursive=True)
                if pat in os.path.basename(p)]
    if not hits:
        sys.exit(f"找不到含 {pat} 的 .db（{DB_DIR}）")
    db = hits[0]
    print("db:", os.path.basename(db), f"{os.path.getsize(db)/1e6:.1f} MB")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    # ---- H-A：lidar_pc 时间戳
    rows = con.execute("SELECT token, timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
    ts = np.array([r[1] for r in rows], dtype=np.float64)
    print(f"\n=== H-A  lidar_pc ===\n  行数 {len(ts)}   "
          f"时长 {(ts[-1]-ts[0])/1e6:.1f} s   名义频率 {len(ts)/((ts[-1]-ts[0])/1e6):.2f} Hz")
    dt = np.diff(ts) / 1e6
    print(f"  dt(s): p50 {np.median(dt):.5f}  p5 {np.percentile(dt,5):.5f}  "
          f"p95 {np.percentile(dt,95):.5f}  min {dt.min():.5f}  max {dt.max():.5f}")
    print(f"  dt 的唯一值（前 8 个）: {np.unique(np.round(dt,6))[:8].tolist()}")
    print(f"  dt == 0 的对数: {(dt == 0).sum()}   （重复时间戳会让行序≠时间序）")
    dec = np.arange(0, len(ts), 2)
    d10 = np.diff(ts[dec]) / 1e6
    print(f"  抽取偶数行后的 dt(s): p50 {np.median(d10):.5f}  min {d10.min():.5f}  "
          f"max {d10.max():.5f}   均匀性 std {d10.std():.5f}")
    print(f"  若 max/min 明显偏离 1，说明每 2 行 ≠ 0.1 s")

    # ---- H-B：lidar_box 重复行
    print("\n=== H-B  lidar_box ===")
    n_rows = con.execute("SELECT COUNT(*) FROM lidar_box").fetchone()[0]
    n_uniq = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT lidar_pc_token, track_token FROM lidar_box)"
    ).fetchone()[0]
    print(f"  行数 {n_rows}    (lidar_pc_token, track_token) 去重后 {n_uniq}   "
          f"重复率 {1 - n_uniq/max(n_rows,1):.4f}")
    colnames = [c[1] for c in con.execute("PRAGMA table_info(lidar_box)")]
    print(f"  列: {colnames}")

    # ---- 取一条振荡轨迹，打印原始 20 Hz 序列
    print("\n=== 逐帧原始序列（前 30 个 20 Hz 帧）===")
    fidx = {r[0]: i for i, r in enumerate(rows)}
    tracks = defaultdict(dict)
    for lpc, trk, x, y, vx, vy, yaw in con.execute(
        "SELECT lidar_pc_token, track_token, x, y, vx, vy, yaw FROM lidar_box"
    ):
        i = fidx.get(lpc)
        if i is not None:
            tracks[trk][i] = (x, y, vx, vy, yaw)
    # 选一个"在偶数帧上几乎全在"的轨道
    best, bestn = None, 0
    for trk, d in tracks.items():
        n = sum(1 for j in dec if j in d)
        if n > bestn:
            best, bestn = trk, n
    print(f"  选中 track {best[:20]}… 覆盖 {bestn}/{len(dec)} 个 10 Hz 帧")
    d = tracks[best]
    present = [j for j in dec if j in d]
    show = present[:14]                    # 该轨道最先出现的 14 个 10 Hz 帧
    print(f"  该轨道首个 10 Hz 帧 index = {show[0] if show else '—'}")
    P = np.array([d[j][:2] for j in show], dtype=np.float64)
    V = np.array([d[j][2:4] for j in show], dtype=np.float64)
    Y = np.array([d[j][4] for j in show], dtype=np.float64)
    T = np.array([ts[j] for j in show], dtype=np.float64) / 1e6
    rel = P - P[0]
    print("  10 Hz 抽样帧 index:", show)
    print("  相对位移 Δp（每步，米）:")
    for k in range(1, len(rel)):
        dtk = T[k] - T[k - 1]
        dd = rel[k] - rel[k - 1]
        print(f"    step {k:2d}  dt={dtk:.4f}s  Δ=({dd[0]:+8.4f}, {dd[1]:+8.4f})"
              f"   |Δ|/dt={np.linalg.norm(dd)/dtk:7.3f} m/s"
              f"   db v=({V[k,0]:+8.3f},{V[k,1]:+8.3f}) |v|={np.linalg.norm(V[k]):7.3f}"
              f"   yaw {np.degrees(Y[k]):+8.2f}°")
    # 直接用 db 里的速度积分 vs 用位置差
    print("\n  对照：db 的 vx,vy 是否与位置差一致（相关系数）:")
    dp = np.diff(rel, axis=0) / 0.1
    if len(dp) > 3:
        for c, nm in enumerate("xy"):
            r = np.corrcoef(dp[:, c], V[1:len(dp)+1, c])[0, 1]
            print(f"    {nm}: corr(Δp/dt, vx{nm}) = {r:+.4f}")
    print("  若相关很低 → 位置列有高频噪声，速度列才是干净的")

    con.close()


if __name__ == "__main__":
    main()
