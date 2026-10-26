"""缓存读取的唯一入口。

历史教训：cache 目录布局从 `logs/<log>.npz` 改成 `npy/<log>/*.npy` 之后，
`verify_cache.py` / `verify_precision.py` / `convert_cache.py` 各自实现了一套
加载逻辑而没同步，于是三个脚本同时以 `IsADirectoryError` 挂掉 —— 而它们恰恰是
用来**证明数据没问题的**脚本，坏掉时没人发现。所以统一到这里。
"""
import json
import os

import numpy as np

KEYS = ("hist", "neigh", "neigh_mask", "future", "tags", "t0", "track_id")


def load_manifest(root):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_norm(root):
    p = os.path.join(root, "norm.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def log_dir(root, entry):
    """entry 来自 manifest（或其等价 dict，含 'file' 字段）→ 日志的 npy 目录。"""
    return os.path.join(root, entry["file"])


def load_log(root, entry, keys=KEYS, mmap=True):
    """读一个 log 的全部数组。返回 {key: ndarray}；缺失的 key 不出现在结果里。"""
    d = log_dir(root, entry)
    if not os.path.isdir(d):
        raise NotADirectoryError(
            f"{d} 不是目录。缓存可能是旧的 npz 布局，或尚未转换；"
            f"先跑 code/data/convert_cache.py --force")
    out = {}
    for k in keys:
        p = os.path.join(d, k + ".npy")
        if os.path.exists(p):
            out[k] = np.load(p, mmap_mode="r" if mmap else None)
    return out


def log_size_mb(root, entry):
    d = log_dir(root, entry)
    if not os.path.isdir(d):
        return 0.0
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1024 / 1024
