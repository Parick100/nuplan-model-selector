"""把整个 nuPlan mini 划分抽成缓存（内存安全版：逐 log 落盘）。

在 WSL 里用 diffusion_planner 的 python 跑：
    /home/parick/ENTER/envs/diffusion_planner/bin/python \
        "/mnt/c/Users/lenovo/Desktop/Homework/Term5/Intro of ML/项目/code/data/build_mini.py" \
        --dir /home/parick/nuplan/dataset/nuplan-v1.1/splits/mini \
        --out /home/parick/dsh_ml/cache

设计要点
- 逐 log 处理并立刻写 <out>/logs/<logfile>.npz，随后释放内存（WSL 仅 ~2 GB 可用）
- 先扫一遍所有 log 的 scenario_tag 建**全局词表**，再正式抽取
- 划分按 **log**（同一 log 不跨份 → 无泄漏，协议 §5），最后写 manifest.json
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import (  # noqa: E402
    DECIM, F, H, M_NEIGH, NEIGH_RADIUS, STRIDE, TARGET_CATEGORY,
    _sample_target, load_boxes, to_target_frame,
)

# 邻居只允许真正的交通参与者（见 neighbours 处注释）
AGENT_CATEGORIES = {"vehicle", "pedestrian", "bicycle"}


def scan_tag_vocab(dbs):
    import sqlite3
    vocab = set()
    for db in dbs:
        try:
            con = sqlite3.connect(db)
            for (t,) in con.execute("SELECT DISTINCT type FROM scenario_tag"):
                vocab.add(str(t))
            con.close()
        except Exception:                                        # noqa: BLE001
            pass
    return sorted(vocab)


def build_index(tracks, dec, n10):
    """一次建成两个索引，把 O(窗口 × 全部轨道) 降到 O(窗口 × 在场的轨道)。

    pres10[f]      : 该 10 Hz 帧在场的 track_token 列表
    frame_agents[f]: 该帧在场的 (track_token, x, y)
    """
    inv = {}
    for j, i20 in enumerate(dec):
        inv[int(i20)] = j
    pres10 = {}
    frame_agents = [[] for _ in range(n10)]
    for trk, d in tracks.items():
        idx = []
        for i20 in d:
            j = inv.get(int(i20))
            if j is not None:
                idx.append(j)
        # 只保留出现 >=2 帧的轨道；frame_agents 必须与 pres10 的键集一致，
        # 否则邻居搜索会 KeyError（曾因单帧轨道踩过这个坑）
        if len(idx) < 2:
            continue
        idx.sort()
        pres10[trk] = np.array(idx, dtype=np.int64)
        for j in idx:
            st = d[dec[j]]
            frame_agents[j].append((trk, st[0], st[1]))
    return pres10, frame_agents


def neighbours_fast(tracks, pres10, frame_agents, dec, t0, x0, y0, th0,
                    radius=50.0, m=10, h=20, exclude_trk=None):
    """与 extract._sample_neighbours 语义一致，但只扫该帧在场的轨道。"""
    r2 = radius * radius
    cands = []
    for trk, x, y in frame_agents[t0]:
        if trk == exclude_trk:
            continue                      # 目标车自身已有独立 token，不占邻居槽
        d2 = (x - x0) ** 2 + (y - y0) ** 2
        if d2 <= r2:
            cands.append((d2, trk))
    cands.sort()

    nb = np.zeros((m, h, 6), dtype=np.float32)
    mask = np.zeros(m, dtype=bool)
    c, s = np.cos(th0), np.sin(th0)
    slot = 0
    for _, trk in cands:
        if slot >= m:
            break
        d = tracks[trk]
        idx_set = set(pres10[trk].tolist())
        if not all((t0 - h + 1 + k) in idx_set for k in range(h)):
            continue
        pts = np.empty((h, 2), dtype=np.float64)
        vels = np.empty((h, 2), dtype=np.float64)
        for k in range(h):
            st = d[dec[t0 - h + 1 + k]]
            pts[k, 0], pts[k, 1] = st[0], st[1]
            vels[k, 0], vels[k, 1] = st[2], st[3]
        xy = to_target_frame(pts, x0, y0, th0)
        nb[slot, :, 0] = xy[:, 0]
        nb[slot, :, 1] = xy[:, 1]
        nb[slot, :, 2] = c * vels[:, 0] + s * vels[:, 1]
        nb[slot, :, 3] = -s * vels[:, 0] + c * vels[:, 1]
        st0 = d[dec[t0]]
        nb[slot, :, 4] = st0[4]
        nb[slot, :, 5] = st0[5]
        mask[slot] = True
        slot += 1
    return nb, mask


def extract_log(db_path, min_fde, vocab):
    """返回 (dict_of_arrays, stats)。内存中只保留单个 log 的样本。"""
    frames, tracks, cat, tags_by_frame, log = load_boxes(db_path)
    n20 = len(frames)
    dec = np.arange(0, n20, DECIM)
    n10 = len(dec)
    vid = {v: k for k, v in enumerate(vocab)}
    T = len(vocab)

    pres10, frame_agents = build_index(tracks, dec, n10)

    # 邻居只保留真正的交通参与者。
    # 审查实测：不过滤时填充槽里只有 33.8% 是车辆，40.3% 是 generic_object、
    # 15.4% 是 traffic_cone —— 10 个槽位里约 2 个被锥桶占掉，交互上下文被污染。
    # Diffusion-Planner 同样把静态物分到独立张量（agent_process.py:69/80）。
    agent_toks = {t for t, c in cat.items() if c in AGENT_CATEGORIES}
    for j in range(n10):
        if frame_agents[j]:
            frame_agents[j] = [a for a in frame_agents[j] if a[0] in agent_toks]

    veh = [t for t, c in cat.items() if c == TARGET_CATEGORY and t in pres10]

    hist_l, fut_l, nb_l, nm_l, tg_l, t0_l, tr_l = [], [], [], [], [], [], []
    n_windows = 0
    t0_start, t0_end = H - 1, n10 - F - 1
    for ti, trk in enumerate(veh):
        idx_set = set(pres10[trk].tolist())
        for t0 in range(t0_start, t0_end + 1, STRIDE):
            if not all(w in idx_set for w in range(t0 - H + 1, t0 + F + 1)):
                continue
            hh, ff = _sample_target(tracks, dec, trk, t0)
            if hh is None:
                continue
            n_windows += 1
            if float(np.linalg.norm(ff[-1])) < min_fde:
                continue
            nb, nbm = neighbours_fast(tracks, pres10, frame_agents, dec, t0,
                                      hh["x0"], hh["y0"], hh["th0"],
                                      radius=NEIGH_RADIUS, m=M_NEIGH, h=H,
                                      exclude_trk=trk)
            tg = np.zeros(T, dtype=np.uint8)
            for t in tags_by_frame.get(dec[t0], []):
                k = vid.get(t)
                if k is not None:
                    tg[k] = 1
            hist_l.append(hh["hist"]); fut_l.append(ff)
            nb_l.append(nb); nm_l.append(nbm); tg_l.append(tg)
            t0_l.append(t0); tr_l.append(ti)

    n = len(hist_l)
    arrs = {
        "hist": np.stack(hist_l).astype(np.float32) if n else np.zeros((0, H, 4), np.float32),
        "future": np.stack(fut_l).astype(np.float32) if n else np.zeros((0, F, 2), np.float32),
        "neigh": np.stack(nb_l).astype(np.float32) if n else np.zeros((0, M_NEIGH, H, 6), np.float32),
        "neigh_mask": np.stack(nm_l).astype(bool) if n else np.zeros((0, M_NEIGH), bool),
        "tags": np.stack(tg_l).astype(np.uint8) if n else np.zeros((0, T), np.uint8),
        "t0": np.array(t0_l, np.int32), "track_id": np.array(tr_l, np.int32),
    }
    stats = {
        "logfile": log[0], "location": log[1], "map_version": log[2],
        "seconds": round(n10 / 10.0, 1), "tracks": len(tracks),
        "vehicle_tracks": len(veh), "windows": n_windows, "kept": n,
        "tracks_with_samples": len(set(tr_l)),
    }
    return arrs, stats


def assign_split(entries, fracs=(0.6, 0.2, 0.2)):
    """按 log 划分，并**按城市分层**（协议 §5：按 log 与城市划分）。

    只在全局上按样本量贪心，会让某个城市整城落进单一 split —— 实测曾出现
    sg-one-north 在模型训练集里 0 样本、us-ma-boston 完全不进入选择器测试集。
    这里在城市之间**轮转**取 log，并按相对缺口分配，保证每份都见到所有城市。
    """
    fracs = np.array(fracs, dtype=np.float64)
    total = float(sum(e["kept"] for e in entries))
    target = np.maximum(fracs * total, 1.0)
    cur = np.zeros(3)
    assign = {}

    by_city = defaultdict(list)
    for i, e in enumerate(entries):
        by_city[e["location"]].append(i)

    # 城市按规模降序处理（大城市先定，小城市补余量）
    order_cities = sorted(by_city, key=lambda c: -sum(entries[i]["kept"] for i in by_city[c]))
    city_has = {c: np.zeros(3) for c in by_city}

    for city in order_cities:
        idxs = sorted(by_city[city], key=lambda i: -entries[i]["kept"])
        for k, i in enumerate(idxs):
            deficit = (target - cur) / target
            empty = [j for j in range(3) if city_has[city][j] == 0]
            remaining = len(idxs) - k
            # 覆盖约束：若该城市在某个 split 里还是 0 样本，且剩下的 log 够填满这些空位，
            # 就优先填空位 —— 否则大城市会靠体量把所有缺口吃掉（实测曾让
            # us-ma-boston 在选择器测试集里 0 样本）
            if empty and remaining >= len(empty):
                j = max(empty, key=lambda x: deficit[x])
            else:
                j = int(np.argmax(deficit))
            assign[i] = j
            cur[j] += entries[i]["kept"]
            city_has[city][j] += entries[i]["kept"]
    return assign, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fde", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dbs = sorted(glob.glob(os.path.join(a.dir, "*.db")))
    if a.limit:
        dbs = dbs[: a.limit]
    if not dbs:
        sys.exit("no .db found")
    logdir = os.path.join(a.out, "logs")
    os.makedirs(logdir, exist_ok=True)

    t_start = time.time()
    print(f"发现 {len(dbs)} 个 log；先扫 scenario_tag 建全局词表 …", flush=True)
    vocab = scan_tag_vocab(dbs)
    print(f"tag 词表 {len(vocab)} 项\n", flush=True)

    entries, failed = [], []
    for i, db in enumerate(dbs):
        name = os.path.basename(db)
        t0 = time.time()
        try:
            arrs, stats = extract_log(db, a.min_fde, vocab)
        except Exception as e:                                   # noqa: BLE001
            failed.append({"logfile": name, "error": repr(e)})
            print(f"[{i+1:2d}/{len(dbs)}] {name[:46]:46s} FAILED {e!r}", flush=True)
            continue
        out_npz = os.path.join(logdir, name.replace(".db", "") + ".npz")
        np.savez_compressed(out_npz, **arrs)
        mb = os.path.getsize(out_npz) // 1024 // 1024
        stats["file"] = os.path.relpath(out_npz, a.out)
        stats["npz_mb"] = mb
        entries.append(stats)
        del arrs
        print(f"[{i+1:2d}/{len(dbs)}] {name[:46]:46s} {stats['seconds']:6.1f}s "
              f"veh {stats['vehicle_tracks']:4d} win {stats['windows']:6d} "
              f"-> {stats['kept']:6d}  {mb:4d}MB  {time.time()-t0:5.1f}s", flush=True)

    if not entries:
        sys.exit("!! 无任何样本")

    counts = [e["kept"] for e in entries]
    assign, cur = assign_split(entries)
    for i, e in enumerate(entries):
        e["split"] = assign[i]

    total = int(sum(counts))
    # 城市 × split 交叉表 —— 证明分层确实生效
    cities = sorted({e["location"] for e in entries})
    city_split = {c: [0, 0, 0] for c in cities}
    for i, e in enumerate(entries):
        city_split[e["location"]][assign[i]] += e["kept"]
    manifest = {
        "n_logs_ok": len(entries), "n_logs_failed": len(failed), "failed": failed,
        "n_samples": total,
        "n_windows_raw": int(sum(e["windows"] for e in entries)),
        "min_fde_filter": a.min_fde,
        "split_by": "log",
        "split_counts": {str(s): int(sum(c for i, c in enumerate(counts) if assign[i] == s))
                         for s in (0, 1, 2)},
        "split_meaning": {"0": "model train", "1": "oracle labels + selector train",
                          "2": "selector test"},
        "tag_vocab": vocab,
        "city_split": city_split,
        "contract": {"hist": [H, 4], "neigh": [M_NEIGH, H, 6], "future": [F, 2]},
        "src_hz": 20, "dst_hz": 10,
        "logs": entries,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(f"log 成功 / 失败 : {len(entries)} / {len(failed)}")
    print(f"原始窗口        : {manifest['n_windows_raw']}")
    print(f"过滤后样本      : {total}"
          f"  (丢弃 {(1 - total / max(manifest['n_windows_raw'], 1)) * 100:.1f}%)")
    print(f"按 log 划分     : {manifest['split_counts']}")
    print("城市 × split 交叉表（每份都应见到所有城市）:")
    for c, v in city_split.items():
        tot = sum(v) or 1
        print(f"   {c:26s} " + "  ".join(f"{x:>7,} ({x/tot*100:4.1f}%)" for x in v)
              + f"   共 {tot:,}")
    print(f"缓存目录        : {a.out}")
    print(f"总耗时          : {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
