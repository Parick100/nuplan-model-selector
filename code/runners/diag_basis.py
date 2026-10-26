"""测量：归一化后的真值轨迹需要多少个 DCT 基函数才能几乎无损表示。

目的：给"输出端低通投影"定 M。
若前 M 个模式就能捕获 >99.9% 能量，那么把输出限制在这 M 维子空间里
**几乎不损失精度，却由构造保证平滑** —— 这是轻量模型最划算的归纳偏置。
"""
import argparse
import json
import os

import numpy as np


def dct_mat(n):
    """DCT-II 正交基，(n, n)，行 = 模式，列 = 时间。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    B = np.cos(np.pi * (i + 0.5) * k / n)
    B[0] *= 1 / np.sqrt(2)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B


def wiggle(t):
    return np.linalg.norm(np.diff(t, n=2, axis=-2), axis=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.root, "manifest.json"), encoding="utf-8"))
    nj = json.load(open(os.path.join(a.root, "norm.json"), encoding="utf-8"))
    mu = np.asarray(nj["mu"], dtype=np.float64)
    sd = np.asarray(nj["sd"], dtype=np.float64)

    logs = [e for e in man["logs"] if e["split"] == 1 and e["kept"] > 0]
    rng = np.random.default_rng(0)
    F, taken = [], 0
    for e in logs:
        f = np.load(os.path.join(a.root, e["file"], "future.npy"), mmap_mode="r")
        k = min(e["kept"], max(1, a.n // len(logs) + 1))
        idx = rng.choice(e["kept"], size=k, replace=False)
        F.append(np.asarray(f[np.sort(idx)]))
        taken += k
        if taken >= a.n:
            break
    F = np.concatenate(F)[: a.n].astype(np.float64)          # (N,80,2) 米制
    Fn = (F - mu) / sd                                       # 归一化
    print(f"样本 {Fn.shape[0]}   真值平滑度 |d²p| = {wiggle(F):.4f} m "
          f"(归一化域 {wiggle(Fn):.5f})")

    B = dct_mat(80)                                          # (80,80)
    C = np.einsum("kt,ntd->nkd", B, Fn)                      # (N,80,2) DCT 系数
    energy = (C ** 2).sum(axis=(0, 2))                       # (80,)
    cum = np.cumsum(energy) / energy.sum()

    print(f"\n{'M':>4} {'能量占比':>10} {'|d²p| 归一化':>14} {'|d²p| 米制':>12} {'vs 真值':>9}")
    w0 = wiggle(F)
    rows = []
    for M in (4, 6, 8, 12, 16, 20, 24, 32, 48, 80):
        Bm = B[:M]
        rec = np.einsum("kt,nkd->ntd", Bm, C[:, :M])          # 投影重建（归一化域）
        rec_m = rec * sd + mu
        w = wiggle(rec_m)
        print(f"{M:>4} {cum[M-1]*100:>9.4f}% {wiggle(rec):>14.6f} {w:>12.4f} {w/w0:>8.2f}x")
        rows.append((M, float(cum[M - 1]), float(w), float(w / w0)))

    print("\n判读：挑一个能量占比 >99.9% 且 |d²p| 明显低于当前模型的 M。")
    print("      当前模型 w=10 时为 4.84 m (84x GT) —— 投影后应远低于此。")


if __name__ == "__main__":
    main()
