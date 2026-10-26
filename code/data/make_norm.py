"""计算逐元素归一化统计量 <root>/norm.json。

**为什么单标量不够（审查实测）**
早先只除以一个全局标量 23.337 m，导致 160 个输出维的 std 从 ≈0.02（未来第 1 步）
到 ≈1.54（第 80 步），**跨度约 5900×**；而生成模型的先验是各向同性单位方差。
4 步流匹配 Euler 会在输出里残留约 0.25–0.34 的单位噪声，对第 1 步而言是真实 std 的
**15 倍** —— 这就是轨迹"坑坑洼洼"的主因，也解释了首个预测点误差 0.91 m
（而真值离原点只有 0.50 m）。

对齐参考实现：Diffusion-Planner 用逐维 `normalization.json`
（`normalizer.py:18-22`，ego/neighbor mean [10,0,0,0] std [20,20,1,1]）。

同时为**锚定槽位**（轨迹第 0 点 = 当前状态，恒为原点）预留 mu=0, sd=1。
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--max-logs", type=int, default=0, help="0 = 全部（大 log 会较慢）")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.root, "manifest.json"), encoding="utf-8"))
    logs = [e for e in man["logs"] if e["split"] == 0 and e["kept"] > 0]
    if a.max_logs:
        logs = sorted(logs, key=lambda e: -e["kept"])[: a.max_logs]

    # 用单精度累加即可（Welford 更稳，但这里样本量足够大）
    n = 0
    s1 = None
    s2 = None
    for e in logs:
        d = os.path.join(a.root, e["file"])
        f = np.load(os.path.join(d, "future.npy"), mmap_mode="r").astype(np.float64)
        if s1 is None:
            s1 = np.zeros_like(f[0])
            s2 = np.zeros_like(f[0])
        s1 += f.sum(axis=0)
        s2 += (f ** 2).sum(axis=0)
        n += f.shape[0]
        del f

    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 1e-12)
    sd = np.sqrt(var)

    out = {
        "mu": mu.tolist(),                       # (80, 2)
        "sd": sd.tolist(),                       # (80, 2)
        "scale": float(sd[-1].mean()),           # 供 hist/neigh 使用的全局标量
        "n_samples": int(n),
        "n_logs": len(logs),
        "definition": "future 的逐 (时间步, 坐标) 均值/标准差；锚点槽位另置 0/1",
        "note": "dataset.py 对 future 做 (f-mu)/sd；指标与可视化再反归一化",
    }
    p = os.path.join(a.root, "norm.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh)

    print(f"{len(logs)} log / {n:,} 样本  ->  {p}")
    print(f"逐元素 std 范围: {sd.min():.4f} .. {sd.max():.4f}"
          f"   (跨度 {sd.max()/max(sd.min(),1e-9):.0f}x)")
    print(f"  未来第 1 步 std  {sd[0].mean():.4f} m   (原全局标量下等效 0.02)")
    print(f"  未来第 80 步 std {sd[-1].mean():.4f} m   (原全局标量下等效 1.54)")
    print(f"  用于 hist/neigh 的全局标量 scale = {out['scale']:.4f} m")
    print("  归一化后：160 个输出维全部 ≈ 单位方差 -> 各向同性先验才成立")


if __name__ == "__main__":
    main()
