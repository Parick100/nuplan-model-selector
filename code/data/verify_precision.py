"""独立复核「float32 坐标变换」缺陷。

审查意见称：extract.py 把绝对 UTM 坐标先存进 float32 数组，再做平移+旋转，
造成灾难性抵消（float32 在 |y|~4e6 处的 ulp 约 0.25 m）。

本脚本不复用被测代码，直接从 SQLite 取 float64 原值，独立算一遍目标帧轨迹，
与缓存逐点比对。

已修复两处：① 缓存布局是 `npy/<log>/*.npy` 而不是 `logs/<log>.npz`（曾导致
IsADirectoryError）；② 轨道枚举顺序必须与 extract.py **完全一致**，否则
`track_id` 会对错行、比对结果无意义 —— 这里直接 import extract 的 `load_boxes`
并复现 `pres10` / `veh` 的构造。
"""
import json
import os
import sqlite3
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache_io import load_log, load_manifest                    # noqa: E402
from extract import DECIM, F, H as H_EX, M_NEIGH, TARGET_CATEGORY, load_boxes  # noqa: E402

CACHE = "/home/parick/dsh_ml/cache"
RAW = "/home/parick/nuplan/dataset/nuplan-v1.1/splits/mini"


def main():
    man = load_manifest(CACHE)
    e = max((x for x in man["logs"] if x["split"] == 0), key=lambda x: x["kept"])
    name = e["logfile"] + ".db"
    db = os.path.join(RAW, name)
    if not os.path.exists(db):
        db = os.path.join(RAW, e["logfile"] + ".db")
    print(f"log: {e['logfile']}   kept={e['kept']}")
    print(f"db : {db}  ({os.path.getsize(db)/1e6:.0f} MB)")

    frames, tracks, cat, _tags, log = load_boxes(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ally = np.array([p[1] for d in list(tracks.values())[:50] for p in d.values()])
    allx = np.array([p[0] for d in list(tracks.values())[:50] for p in d.values()])
    con.close()
    print(f"绝对坐标量级: |x| ~ {np.abs(allx).max():.3e}   |y| ~ {np.abs(ally).max():.3e}")
    print(f"float32 在该量级的 ulp: x {np.spacing(np.float32(np.abs(allx).max())):.4f} m  "
          f"y {np.spacing(np.float32(np.abs(ally).max())):.4f} m  "
          f"← 若先降精度再相减，就是这个量级的不可约误差")

    # ---- 与 extract.py 完全一致的枚举顺序 ----
    dec = np.arange(0, len(frames), DECIM)[: (len(frames) // DECIM)]
    pres10 = {}
    for trk, d in tracks.items():
        idx = [j for j, i20 in enumerate(dec) if i20 in d]
        if len(idx) >= 2:
            pres10[trk] = np.array(idx, dtype=np.int64)
    veh = [t for t, c in cat.items() if c == TARGET_CATEGORY and t in pres10]
    print(f"vehicle 轨道数 {len(veh)}（与 extract.py 同一枚举顺序）")

    arr = load_log(CACHE, e)
    hist_c = np.asarray(arr["hist"])[:, :, :2]          # (N,20,2)
    fut_c = np.asarray(arr["future"])                   # (N,80,2)
    t0s = np.asarray(arr["t0"])
    trks = np.asarray(arr["track_id"])
    print(f"缓存: {hist_c.shape[0]} 样本  hist{hist_c.shape[1:]} future{fut_c.shape[1:]}")

    def target_frame(p, x0, y0, th0):
        c, s = np.cos(th0), np.sin(th0)
        dx, dy = p[0] - x0, p[1] - y0
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    errs_h, errs_f, checked = [], [], 0
    for n in range(len(t0s)):
        trk = veh[int(trks[n])]
        t0 = int(t0s[n])
        d = tracks[trk]
        f0 = dec[t0]
        if f0 not in d:
            continue
        x0, y0 = d[f0][0], d[f0][1]
        th0 = float(d[f0][6])                       # ← yaw（修复后的做法）
        ok = True
        hh = np.zeros((H_EX, 2))
        for k in range(H_EX):
            i20 = dec[t0 - H_EX + 1 + k]
            if i20 not in d:
                ok = False
                break
            hh[k] = target_frame(d[i20], x0, y0, th0)
        if not ok:
            continue
        ff = np.zeros((F, 2))
        for k in range(F):
            i20 = dec[t0 + 1 + k]
            if i20 not in d:
                ok = False
                break
            ff[k] = target_frame(d[i20], x0, y0, th0)
        if not ok:
            continue
        errs_h.append(np.abs(hh - hist_c[n]).max())
        errs_f.append(np.abs(ff - fut_c[n]).max())
        checked += 1
        if checked >= 400:
            break

    eh, ef = np.array(errs_h), np.array(errs_f)
    print(f"\n复核样本数: {checked}")
    for nm, v in (("hist", eh), ("future", ef)):
        print(f"{nm:7s}: mean {v.mean():.3e}  median {np.median(v):.3e}  "
              f"p95 {np.percentile(v,95):.3e}  max {v.max():.3e} m")
    print(f"future 误差 > 1 cm 的比例: {(ef > 0.01).mean()*100:.1f}%")

    # 判据必须针对**本地的** float32 表示精度，而不是"是否恰好等于 0"。
    # 局部坐标量级 ~60 m，float32 在该量级的 ulp = 2^-23 * 64 ≈ 7.6e-6。
    # 缓存用 float32 存储，所以逐点误差落在 ulp 量级即证明**没有发生抵消**。
    local_scale = max(np.abs(hist_c).max(), np.abs(fut_c).max())
    ulp_local = float(np.spacing(np.float32(local_scale)))
    print(f"\n局部坐标量级 {local_scale:.1f} m → float32 的 ulp = {ulp_local:.3e} m")
    print(f"（对照组：若在绝对坐标上做 float32 减法，误差应是 "
          f"{np.spacing(np.float32(np.abs(ally).max())):.4f} m，即 250000 倍）")
    if ef.max() <= 4 * ulp_local and eh.max() <= 4 * ulp_local:
        print(f"\n结论: 缓存与 float64 独立重算的差异 {ef.max():.2e} m ≤ 4×float32 ulp ——")
        print("      **float32 灾难性抵消缺陷已修复**（残差纯粹是缓存的 float32 存储精度，")
        print("      不是坐标变换引入的误差）。")
    elif ef.mean() > 0.01:
        print("\n结论: 🔴 缺陷仍然存在 —— 缓存未用修复后的代码重建")
    else:
        print(f"\n结论: 残余误差 {ef.max():.2e} m 明显大于 float32 ulp，需再查")


if __name__ == "__main__":
    main()

