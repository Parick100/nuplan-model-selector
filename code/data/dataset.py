"""nuPlan mini 轨迹预测数据集（逐 log 惰性加载，mmap 零拷贝，逐元素归一化）。

缓存布局：
    <root>/manifest.json      每 log 的文件、样本数、split
    <root>/norm.json          mu(80,2) / sd(80,2) / 全局 scale
    <root>/npy/<log>/*.npy    hist / neigh / neigh_mask / future / tags / t0 / track_id

划分语义（协议 §5，按 log 且按城市）：
    split 0 = 模型训练 · split 1 = oracle 标注 + 选择器训练 · split 2 = 选择器测试

归一化（见 make_norm.py）：
    future -> (f - mu) / sd      逐 (时间步, 坐标)，使 160 维输出近似各向同性单位方差
    hist / neigh -> / scale      仅作编码器输入，全局标量足够
"""
import json
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:                                                # noqa: BLE001
    torch = None

    class Dataset:                                               # type: ignore
        pass

KEYS = ("hist", "neigh", "neigh_mask", "future", "tags", "t0", "track_id")


class MiniTrajDataset(Dataset):
    def __init__(self, root, split, cache_logs=4, include_tags=True, normalize=True):
        self.root = root
        man = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
        self.manifest = man
        self.vocab = man["tag_vocab"]
        self.tag_dim = len(self.vocab)
        self.include_tags = include_tags
        self.logs = [e for e in man["logs"] if e["split"] == split and e["kept"] > 0]

        self.scale = 1.0
        self.mu = None
        self.sd = None
        npath = os.path.join(root, "norm.json")
        if normalize and os.path.exists(npath):
            with open(npath, encoding="utf-8") as fh:
                nj = json.load(fh)
            self.scale = float(nj.get("scale", 1.0))
            if "mu" in nj:                       # 逐元素统计（新格式）
                self.mu = np.asarray(nj["mu"], dtype=np.float32)
                self.sd = np.asarray(nj["sd"], dtype=np.float32)

        self.index = []
        for li, e in enumerate(self.logs):
            self.index.extend([(li, i) for i in range(e["kept"])])

        self.cache_logs = max(1, cache_logs)
        self._cache = {}
        self._lru = []

    # ---- mmap 载入单个 log（零拷贝；压缩 npz 曾导致 6 s/step）----
    def _arrays(self, li):
        if li in self._cache:
            self._lru.remove(li)
            self._lru.append(li)
            return self._cache[li]
        d = os.path.join(self.root, self.logs[li]["file"])
        arrs = {}
        for k in KEYS:
            p = os.path.join(d, k + ".npy")
            if os.path.exists(p):
                arrs[k] = np.load(p, mmap_mode="r")
        self._cache[li] = arrs
        self._lru.append(li)
        while len(self._lru) > self.cache_logs:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return arrs

    # ---- 反归一化（指标/可视化用）----
    def denorm_future(self, f):
        """(..., 80, 2) 归一化空间 -> 米制。"""
        if self.mu is None:
            return f
        return f * self.sd + self.mu

    def log_of(self, i):
        """样本 i 属于哪个 log（做 log 级聚类统计用）。"""
        return self.logs[self.index[i][0]]["logfile"]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        li, j = self.index[i]
        a = self._arrays(li)
        hist = a["hist"][j] / self.scale
        neigh = a["neigh"][j] / self.scale
        fut = a["future"][j]
        if self.mu is not None:
            fut = (fut - self.mu) / self.sd
        out = {
            "hist": torch.from_numpy(np.array(hist, dtype=np.float32)),
            "neigh": torch.from_numpy(np.array(neigh, dtype=np.float32)),
            "neigh_mask": torch.from_numpy(np.array(a["neigh_mask"][j])),
            "future": torch.from_numpy(np.array(fut, dtype=np.float32)),
            "log_id": li,
        }
        if self.include_tags:
            out["tags"] = torch.from_numpy(a["tags"][j].astype(np.float32))
        return out

    def stats(self):
        return {"n": len(self), "logs": len(self.logs), "tag_dim": self.tag_dim,
                "per_element_norm": self.mu is not None, "scale": self.scale}

    def iter_logs(self):
        for li, e in enumerate(self.logs):
            yield e, self._arrays(li)


def describe_split(root, split):
    ds = MiniTrajDataset(root, split)
    print(f"split {split}: {ds.stats()}")
    return ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    a = ap.parse_args()
    for s in (0, 1, 2):
        describe_split(a.root, s)
