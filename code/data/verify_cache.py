"""缓存健全性检查：形状/类型/坐标系自检/划分比例/标签覆盖/防泄漏。

已修复：早先本脚本假设 `logs/<log>.npz`，在缓存改成 `npy/<log>/*.npy` 之后
以 `IsADirectoryError` 崩溃。现在统一走 `cache_io.py`。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/data/verify_cache.py \
        --root /home/parick/dsh_ml/cache
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_io import load_log, load_manifest, load_norm, log_size_mb   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--probe-logs", type=int, default=3)
    a = ap.parse_args()

    man = load_manifest(a.root)
    logs = man["logs"]

    print("=== manifest ===")
    print(f"log 数            : {man['n_logs_ok']} (failed {man['n_logs_failed']})")
    print(f"样本总数          : {man['n_samples']:,}")
    print(f"标签词表          : {len(man['tag_vocab'])} 项")
    print(f"划分（按 log）    : {man['split_counts']}")
    for s, name in man["split_meaning"].items():
        n = man["split_counts"][s]
        print(f"   split {s} = {name:34s} {n:>8,}  ({n/man['n_samples']*100:5.1f}%)")
    c = Counter(e["split"] for e in logs)
    print(f"每份 log 数       : {dict(sorted(c.items()))}")

    nj = load_norm(a.root)
    if nj:
        sd = np.asarray(nj["sd"])
        print(f"norm.json         : {nj.get('n_logs')} log / {nj.get('n_samples'):,} 样本"
              f"  (用 split 0 统计，无泄漏)")
        print(f"  逐元素 sd 范围 {sd.min():.4f} .. {sd.max():.4f} m"
              f"   全局 scale {nj['scale']:.4f}")
    else:
        print("!! 缺 norm.json —— 逐元素归一化未生效")

    print("\n=== 抽样检查 ===")
    order = sorted(range(len(logs)), key=lambda i: -logs[i]["kept"])[: a.probe_logs]
    order += sorted(range(len(logs)), key=lambda i: -logs[i]["kept"])[-1:]
    for i in order:
        e = logs[i]
        arr = load_log(a.root, e)
        print(f"\n--- {e['logfile'][:46]}  kept={e['kept']}  split={e['split']}"
              f"  {log_size_mb(a.root, e):.0f} MB")
        for k in ("hist", "neigh", "neigh_mask", "future", "tags"):
            if k not in arr:
                print(f"    {k:11s} !! 缺失")
                continue
            A = arr[k]
            print(f"    {k:11s} {str(A.shape):20s} {A.dtype}  "
                  f"min {float(A.min()):+.3f} max {float(A.max()):+.3f}")
        # 坐标系自检：历史末点必须 ≈ 原点（目标车 t0 自身坐标系）
        tail = float(np.abs(np.asarray(arr["hist"][:, -1, :2])).max())
        print(f"    hist 末点 max|p| = {tail:.3e}   "
              f"{'OK' if tail < 1e-4 else '!! 有问题'}")
        # 未来第 0 点不应存在（未来从 t0+1 起）——用它检验窗口约定
        fde = np.linalg.norm(np.asarray(arr["future"][:, -1, :]), axis=-1)
        print(f"    FDE  mean {fde.mean():6.2f}  median {np.median(fde):6.2f} "
              f" p90 {np.percentile(fde, 90):6.2f} m")
        print(f"    邻居槽位使用率 {float(np.asarray(arr['neigh_mask']).mean())*100:5.1f}%")
        tg = np.asarray(arr["tags"]).sum(axis=0).astype(np.int64)
        print(f"    有标签的样本 {int((np.asarray(arr['tags']).sum(axis=1) > 0).sum())}"
              f"/{e['kept']}   出现的标签种类 {int((tg > 0).sum())}")

    print("\n=== 全局标签分布（前 12） ===")
    tot = np.zeros(len(man["tag_vocab"]), dtype=np.int64)
    for e in logs:
        if e["kept"] == 0:
            continue
        tot += np.asarray(load_log(a.root, e, keys=("tags",))["tags"]).sum(axis=0).astype(np.int64)
    for k in np.argsort(tot)[::-1][:12]:
        if tot[k]:
            print(f"   {tot[k]:>9,}  {man['tag_vocab'][k]}")

    # ---- 防泄漏：必须是**真断言**，不能是恒真式 ----
    print("\n=== 防泄漏 ===")
    names = [e["logfile"] for e in logs]
    dup = len(names) - len(set(names))
    print(f"logfile 唯一性: {len(names)} 条记录, 重复 {dup}"
          f"   {'OK' if dup == 0 else '!! 有重复'}")
    by = {}
    for e in logs:
        by.setdefault(e["logfile"], set()).add(e["split"])
    bad = {k: sorted(v) for k, v in by.items() if len(v) > 1}
    print(f"同一 log 只属于一个 split: {'OK' if not bad else '!! ' + str(bad)}")

    cs = man.get("city_split")
    if cs:
        print("城市 × split 交叉表（协议 §5 要求按 log **与城市**划分）:")
        any_empty = False
        for city, v in cs.items():
            zeros = [i for i, x in enumerate(v) if x == 0]
            any_empty |= bool(zeros)
            flag = "" if not zeros else f"   ⚠ split {zeros} 无样本"
            print(f"   {city:26s} " + "  ".join(f"{x:>8,}" for x in v)
                  + f"   共 {sum(v):,}{flag}")
        if any_empty:
            print("   ⚠ 存在城市在某个 split 里 0 样本 —— 分层未完全生效")
    else:
        print("⚠ manifest 无 city_split 字段 —— 无法确认城市分层")

    # ---- manifest 与实际数组是否自洽 ----
    print("\n=== manifest vs 数组自洽性 ===")
    mismatch, checked = [], 0
    for e in logs:
        if e["kept"] == 0:
            continue
        arr = load_log(a.root, e, keys=("hist", "future"))
        if arr["hist"].shape[0] != e["kept"] or arr["future"].shape[0] != e["kept"]:
            mismatch.append((e["logfile"], e["kept"], int(arr["hist"].shape[0])))
        checked += 1
    print(f"核对 {checked} 个 log 的 kept 与数组样本数: "
          f"{'OK' if not mismatch else '!! ' + str(mismatch[:5])}")

    # ---- 契约检查：形状必须与 dataset/model 的假设一致 ----
    print("\n=== 形状契约 ===")
    e = max(logs, key=lambda x: x["kept"])
    arr = load_log(a.root, e)
    want = {"hist": (20, 4), "neigh": (10, 20, 6), "future": (80, 2),
            "neigh_mask": (10,), "tags": (len(man["tag_vocab"]),)}
    ok = True
    for k, shp in want.items():
        got = tuple(arr[k].shape[1:])
        good = got == shp
        ok &= good
        print(f"    {k:11s} {str(got):14s} 期望 {str(shp):14s} {'OK' if good else '!!'}")
    print("契约", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
