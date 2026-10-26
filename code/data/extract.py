"""W1 — nuPlan mini 抽取管线（纯 stdlib sqlite3，不依赖 nuplan-devkit）。

产出：每个预测样本
    hist       (20, 4)      目标车过去 2 s：x, y, vx, vy（目标车 t0 自身坐标系）
    neigh      (M, 20, 6)   邻居车过去 2 s：x, y, vx, vy, w, l（同上坐标系）
    neigh_mask (M,)         bool，该邻居槽位是否有效
    future     (80, 2)      目标车未来 8 s：x, y（同上坐标系）
    tags       (T,)         uint8 多热，来自 scenario_tag
    log_id     int

用法：
    python code/data/extract.py --db <log.db> --out code/data/cache
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

# ---- 任务常量（与 EXPERIMENTS_实验协议.md §1 对齐）----
SRC_HZ = 20          # lidar_pc 实测频率
DST_HZ = 10          # 目标频率
DECIM = SRC_HZ // DST_HZ
H = 20               # 历史帧数（2 s @10 Hz）
F = 80               # 未来帧数（8 s @10 Hz）
WIN = H + F          # 10 s @10 Hz
M_NEIGH = 10         # 邻居槽位数
NEIGH_RADIUS = 50.0  # m
STRIDE = 5           # t0 采样步长（0.5 s @10 Hz）
TARGET_CATEGORY = "vehicle"


def load_boxes(db_path):
    """读 lidar_pc 帧序 + lidar_box 状态，返回 (frames, tracks)."""
    con = sqlite3.connect(db_path)
    frames = [t for (t,) in con.execute("SELECT token FROM lidar_pc ORDER BY timestamp")]
    fidx = {t: i for i, t in enumerate(frames)}

    tracks = defaultdict(dict)   # track_token -> {20Hz idx: (x,y,vx,vy,w,l,yaw)}
    for lpc, trk, x, y, vx, vy, w, l, yaw in con.execute(
        "SELECT lidar_pc_token, track_token, x, y, vx, vy, width, length, yaw FROM lidar_box"
    ):
        i = fidx.get(lpc)
        if i is not None:
            tracks[trk][i] = (x, y, vx, vy, w, l, yaw)

    # 轨道类别
    t2c = {t: c for t, c in con.execute("SELECT token, category_token FROM track")}
    c2n = {t: n for t, n in con.execute("SELECT token, name FROM category")}
    cat = {t: c2n.get(c, "?") for t, c in t2c.items()}

    # scenario_tag：按 lidar_pc 聚合
    tags_by_frame = defaultdict(list)
    for lpc, typ in con.execute("SELECT lidar_pc_token, type FROM scenario_tag"):
        i = fidx.get(lpc)
        if i is not None:
            tags_by_frame[i].append(str(typ))

    log = con.execute("SELECT logfile, location, map_version FROM log").fetchone()
    con.close()
    return frames, tracks, cat, tags_by_frame, log


def to_target_frame(pts, x0, y0, th0):
    """全局坐标 -> 目标车 t0 自身坐标系（平移 + 旋转 -th0）。

    ⚠ 必须在 float64 下做减法。
    nuPlan 使用**绝对 UTM 坐标**（实测 |x|≈6.6e5、|y|≈4.0e6），float32 在该量级的
    ulp 分别是 0.0625 m 与 **0.25 m**。若先把绝对坐标降到 float32 再相减，会产生
    灾难性抵消：真值被量化到与全局网格对齐的 0.25 m 栅格上，给每条轨迹引入约
    0.17 m 的不可约误差（实测 mean 0.173 / max 0.241 m，100% 样本 >1 cm）。
    由于局部坐标只有几十米，float64 算完再降为 float32 完全无损。
    """
    pts = np.asarray(pts, dtype=np.float64)          # 兜底：防调用方传 float32
    x0, y0, th0 = float(x0), float(y0), float(th0)
    c, s = np.cos(th0), np.sin(th0)
    dx = pts[..., 0] - x0
    dy = pts[..., 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def build(db_path, out_dir, min_fde=2.0):
    frames, tracks, cat, tags_by_frame, log = load_boxes(db_path)
    n20 = len(frames)
    dec = np.arange(0, n20, DECIM)[: (n20 // DECIM)]      # 10 Hz 帧 -> 20 Hz 帧
    n10 = len(dec)
    print(f"log           : {log[0]} @ {log[1]} (map {log[2]})")
    print(f"frames 20 Hz  : {n20}   ->  10 Hz : {n10}   ({n10/10:.1f} s)")
    print(f"tracks        : {len(tracks)}")

    # 10 Hz 视角下的轨道存在性
    pres10 = {}
    for trk, d in tracks.items():
        idx = [j for j, i20 in enumerate(dec) if i20 in d]
        if len(idx) >= 2:
            pres10[trk] = np.array(idx, dtype=np.int64)

    tag_vocab = sorted({t for ts in tags_by_frame.values() for t in ts})
    tag_id = {t: k for k, t in enumerate(tag_vocab)}
    print(f"scenario tags : {len(tag_vocab)}")

    veh = [t for t, c in cat.items() if c == TARGET_CATEGORY and t in pres10]

    rows = []
    t0_start = H - 1
    t0_end = n10 - F - 1
    n_windows = 0

    for ti, trk in enumerate(veh):
        idx = pres10[trk]
        idx_set = set(idx.tolist())
        # 窗口 = [t0-H+1, t0+F]：历史末点即 t0（当前帧），未来从 t0+1 起
        for t0 in range(t0_start, t0_end + 1, STRIDE):
            win = range(t0 - H + 1, t0 + F + 1)
            if not all(w in idx_set for w in win):
                continue
            hist_l, fut_l = _sample_target(tracks, dec, trk, t0)
            if hist_l is None:
                continue
            n_windows += 1
            fde = float(np.linalg.norm(fut_l[-1]))
            if fde < min_fde:
                continue                      # 近静止目标：不可预测性太低，会淹没指标
            nb, nbm = _sample_neighbours(tracks, cat, pres10, dec, t0,
                                         hist_l["x0"], hist_l["y0"], hist_l["th0"],
                                         exclude_trk=trk)
            tg = np.zeros(len(tag_vocab), dtype=np.uint8)
            for t in tags_by_frame.get(dec[t0], []):
                tg[tag_id[t]] = 1
            rows.append({
                "hist": hist_l["hist"], "future": fut_l, "neigh": nb,
                "nmask": nbm, "tags": tg, "t0": int(t0), "track_id": ti, "fde": fde,
            })

    print(f"\n窗口漏斗：全部 {n_windows}  ->  FDE >= {min_fde:.1f} m 后 {len(rows)}"
          f"   (丢弃 {n_windows - len(rows)} = {(n_windows - len(rows)) / max(n_windows,1) * 100:.1f}%)")
    if not rows:
        sys.exit("!! 过滤后无样本，请调低 --min-fde")

    hist = np.stack([r["hist"] for r in rows])
    future = np.stack([r["future"] for r in rows])
    neigh = np.stack([r["neigh"] for r in rows])
    nmask = np.stack([r["nmask"] for r in rows])
    tags = np.stack([r["tags"] for r in rows])
    logid = np.zeros(len(rows), dtype=np.int32)
    t0_arr = np.array([r["t0"] for r in rows], dtype=np.int32)
    trk_arr = np.array([r["track_id"] for r in rows], dtype=np.int32)

    # 按 track 划分，并让三份的**样本数**大致 60/20/20（单 log 时的退化方案，见协议 §5）
    split = np.full(len(rows), -1, dtype=np.int8)
    uniq, counts = np.unique(trk_arr, return_counts=True)
    target = np.array([0.6, 0.2, 0.2]) * len(rows)
    cur = np.zeros(3, dtype=np.float64)
    order = uniq[np.argsort(-counts)]
    for t in order:
        j = int(np.argmax(target - cur))
        split[trk_arr == t] = j
        cur[j] += counts[np.where(uniq == t)[0][0]]
    print(f"\n划分（按 track，目标 60/20/20）：实际 {cur.astype(int).tolist()}")
    print(f"含样本 track 数 {len(uniq)}   单 track 最多 {int(counts.max())} 条"
          f"   中位 {int(np.median(counts))} 条")
    if cur.min() < 0.08 * len(rows):
        print("⚠ 最小划分样本过少：按 track 划分时少数 track 独占多数样本，"
              "无法在保持无泄漏的前提下摊平。→ 需要更多 log。")

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "samples.npz")
    np.savez_compressed(
        out_npz,
        hist=hist.astype(np.float32),
        neigh=neigh.astype(np.float32),
        neigh_mask=nmask.astype(bool),
        future=future.astype(np.float32),
        tags=tags.astype(np.uint8),
        log_id=logid,
        track_id=trk_arr,
        t0=t0_arr,
        split=split,
    )

    meta = {
        "logs": [{"logfile": log[0], "location": log[1], "map_version": log[2],
                  "src_hz": SRC_HZ, "frames": n20}],
        "tag_vocab": tag_vocab,
        "min_fde_filter": min_fde,
        "n_windows_raw": int(n_windows),
        "n_samples": len(rows),
        "n_target_tracks": len(veh),
        "split_counts": {str(s): int((split == s).sum()) for s in (0, 1, 2)},
        "split_meaning": {"0": "model train", "1": "oracle labels + selector train",
                          "2": "selector test"},
        "contract": {"hist": [H, 4], "neigh": [M_NEIGH, H, 6], "future": [F, 2]},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n=== 产出 ===")
    print(f"samples       : {len(rows)}")
    print(f"target tracks : {len(veh)}")
    print(f"split 0/1/2   : {meta['split_counts']}")
    print(f"npz           : {out_npz}  ({os.path.getsize(out_npz)//1024} KB)")
    _report(future, hist, nmask, tags, tag_vocab)


def _sample_target(tracks, dec, trk, t0):
    d = tracks[trk]
    f0 = dec[t0]
    st0 = d[f0]
    x0, y0 = st0[0], st0[1]
    # 朝向直接取 lidar_box.yaw。
    # 早先版本用相邻两帧的位置差 atan2(y0-ya, x0-xa) 估计朝向，这在车辆刚起步
    # （0.1 s 位移 < 5 cm，占 15.7%）时病态：实测中位误差 66.6°、43% 超过 90°，
    # 位移恰为 0 时 arctan2(0,0)=0 更会把坐标系静默退化成全局系。
    # yaw 与位置差分方向中位仅差 0.88°，是正确且稳定的选择。
    th0 = float(st0[6])
    hist = np.zeros((H, 4), dtype=np.float64)      # float64！见 to_target_frame 注释
    for k in range(H):
        i20 = dec[t0 - H + 1 + k]
        if i20 not in d:
            return None, None
        st = d[i20]
        hist[k] = (st[0], st[1], st[2], st[3])
    fut = np.zeros((F, 2), dtype=np.float64)
    for k in range(F):
        i20 = dec[t0 + 1 + k]
        if i20 not in d:
            return None, None
        fut[k] = (d[i20][0], d[i20][1])

    hist_xy = to_target_frame(hist[:, :2], x0, y0, th0)
    c, s = np.cos(th0), np.sin(th0)
    vx_r = c * hist[:, 2] + s * hist[:, 3]
    vy_r = -s * hist[:, 2] + c * hist[:, 3]
    hist_t = np.stack([hist_xy[:, 0], hist_xy[:, 1], vx_r, vy_r], axis=-1).astype(np.float32)
    fut_t = to_target_frame(fut, x0, y0, th0).astype(np.float32)
    return {"hist": hist_t, "x0": x0, "y0": y0, "th0": th0}, fut_t


def _sample_neighbours(tracks, cat, pres10, dec, t0, x0, y0, th0, exclude_trk=None):
    f0 = dec[t0]
    cands = []
    for trk, idx in pres10.items():
        if trk == exclude_trk:
            # 目标车自身距离为 0，必然排在第一槽 —— 但它已有独立 token，
            # 重复放进邻居槽会浪费一个槽位，也会让 hist 与 neigh[0] 语义重叠
            continue
        if f0 not in tracks[trk]:
            continue
        x, y = tracks[trk][f0][0], tracks[trk][f0][1]
        d2 = (x - x0) ** 2 + (y - y0) ** 2
        if d2 <= NEIGH_RADIUS ** 2:
            cands.append((d2, trk))
    cands.sort()

    nb = np.zeros((M_NEIGH, H, 6), dtype=np.float32)
    mask = np.zeros(M_NEIGH, dtype=bool)
    slot = 0
    for _, trk in cands:
        if slot >= M_NEIGH:
            break
        d = tracks[trk]
        idx_set = set(pres10[trk].tolist())
        if not all((t0 - H + 1 + k) in idx_set for k in range(H)):
            continue
        pts = np.array([d[dec[t0 - H + 1 + k]][:2] for k in range(H)], dtype=np.float64)
        vels = np.array([d[dec[t0 - H + 1 + k]][2:4] for k in range(H)], dtype=np.float64)
        xy = to_target_frame(pts, x0, y0, th0)
        c, s = np.cos(th0), np.sin(th0)
        vx_r = c * vels[:, 0] + s * vels[:, 1]
        vy_r = -s * vels[:, 0] + c * vels[:, 1]
        st0 = d[dec[t0]]
        w, l = st0[4], st0[5]
        nb[slot, :, 0] = xy[:, 0]
        nb[slot, :, 1] = xy[:, 1]
        nb[slot, :, 2] = vx_r
        nb[slot, :, 3] = vy_r
        nb[slot, :, 4] = w
        nb[slot, :, 5] = l
        mask[slot] = True
        slot += 1
    return nb, mask


def _report(future, hist, nmask, tags, vocab):
    fde = np.linalg.norm(future[:, -1, :], axis=-1)
    ade = np.linalg.norm(future, axis=-1).mean(axis=-1)
    print("\n=== 健全性检查 ===")
    print(f"hist 末点应≈原点  max|p| = {np.abs(hist[:, -1, :2]).max():.3e}"
          f"   {'OK' if np.abs(hist[:, -1, :2]).max() < 1e-4 else '!! 有问题'}")
    print("   注意：该检查是**必要但非充分**的 —— 目标车 t0 位置与自身相减恒为 0，"
          "因此它在 float32 下也会通过。精度问题请用 data/verify_precision.py 复核。")
    print(f"邻居槽位使用率   {nmask.mean()*100:.1f}%")
    print(f"\nfuture FDE: mean {fde.mean():.2f}  median {np.median(fde):.2f}  "
          f"p90 {np.percentile(fde, 90):.2f}  max {fde.max():.2f} m")
    print(f"future ADE: mean {ade.mean():.2f}  median {np.median(ade):.2f} m")
    for thr in (0.5, 1.0, 2.0):
        frac = (fde < thr).mean()
        print(f"  FDE < {thr:>3.1f} m : {frac*100:5.1f}%  (近静止/易样本)")
    print("场景标签分布 top8:")
    tot = tags.sum(axis=0).astype(np.int64)      # 注意：uint8 取负会回绕，必须先转 int
    order = np.argsort(-tot)
    shown = 0
    for k in order:
        if tot[k]:
            print(f"   {int(tot[k]):5d} / {len(tags):5d}  {vocab[k]}")
            shown += 1
            if shown >= 8:
                break
    if shown == 0:
        print("   (无)")

    mov = np.linalg.norm(hist[:, -1, :2] - hist[:, 0, :2], axis=-1)
    print(f"\n历史 2 s 位移: mean {mov.mean():.2f} m   —— 目标确实是运动中的")


def split_by_log(log_id, seed=0):
    """按 log 划分。当前仅 1 个 log，故退化为按 track 划分（见文档 §5）。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(log_id)
    order = rng.permutation(uniq)
    n = len(order)
    a = order[: max(1, int(n * 0.6))]
    b = order[max(1, int(n * 0.6)): max(2, int(n * 0.8))]
    c = order[max(2, int(n * 0.8)):]
    return a, b, c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=os.path.join("code", "data", "cache"))
    ap.add_argument("--min-fde", type=float, default=2.0,
                    help="丢弃未来 8 s 位移小于该值的目标（近静止样本会淹没指标）")
    a = ap.parse_args()
    if not os.path.exists(a.db):
        sys.exit(f"DB not found: {a.db}")
    build(a.db, a.out, min_fde=a.min_fde)
