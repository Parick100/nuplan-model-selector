"""诊断：为什么单样本过拟合 loss 卡在 0.107、minADE 却是 9 m？

假设 H1：低通投影把真值本身打掉了（loss 地板 = ||gt − P gt||²）。
假设 H2：归一化 sd 形状病态，让"归一化误差小"与"米制误差大"同时成立。

不依赖模型：直接对真值做代数运算就能判。
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))

from dataset import MiniTrajDataset                              # noqa: E402
from model import dct_lowpass                                    # noqa: E402

ROOT = "/home/parick/dsh_ml/cache"

nj = json.load(open(os.path.join(ROOT, "norm.json"), encoding="utf-8"))
mu = np.asarray(nj["mu"], dtype=np.float64)      # (80,2)
sd = np.asarray(nj["sd"], dtype=np.float64)
print("=== norm.json ===")
print("scale:", nj.get("scale"))
for c, name in enumerate("xy"):
    print(f"  {name}: mu {mu[0,c]:8.4f} -> {mu[-1,c]:8.3f}   "
          f"sd {sd[0,c]:8.5f} -> {sd[-1,c]:8.3f}   "
          f"min/max sd {sd[:,c].min():.5f} / {sd[:,c].max():.3f}")
print("  sd 全 160 维的分位: " +
      "  ".join(f"p{q}={np.percentile(sd, q):.4f}" for q in (0, 5, 25, 50, 75, 95, 100)))

ds = MiniTrajDataset(ROOT, 0)
e = max(ds.logs, key=lambda x: x["kept"])
li = ds.logs.index(e)
# ⚠ 必须用**全局**下标：`ds.index` 的元素是 (log下标, log内行号)，
# 只取行号再喂给 ds[i] 会全部落到数据集开头的 log 0（审查抓到的 bug）。
rows = [k for k, (a, _j) in enumerate(ds.index) if a == li]
rng = np.random.default_rng(0)
pick = rng.choice(rows, size=256, replace=False)
logs_hit = {ds.log_of(int(i)) for i in pick}
assert len(logs_hit) == 1 and e["logfile"] in logs_hit, f"抽样跑偏: {logs_hit}"
gt_n = np.stack([np.asarray(ds[int(i)]["future"]) for i in pick])     # (N,80,2) 归一化
gt_m = ds.denorm_future(gt_n)                                        # (N,80,2) 米

print(f"\n=== 真值（{len(pick)} 条，log={e['logfile'][:36]}）===")
print(f"  米制: |p1| {np.linalg.norm(gt_m[:,0],axis=-1).mean():7.3f}  "
      f"|p40| {np.linalg.norm(gt_m[:,39],axis=-1).mean():7.3f}  "
      f"|p80| {np.linalg.norm(gt_m[:,-1],axis=-1).mean():7.3f} m")
print(f"  归一化: 逐元素 std 均值 {gt_n.std(axis=0).mean():.5f}  "
      f"（若归一化正确应 ≈1.0）")
print(f"  归一化: 逐元素 std 范围 {gt_n.std(axis=0).min():.5f} .. {gt_n.std(axis=0).max():.5f}")

# ---- H1：米制空间的低通投影对真值做了什么（**逐坐标**沿时间轴投影）
P = dct_lowpass(80, 24).numpy().astype(np.float64)
proj_m = np.einsum("ij,bjc->bic", P, gt_m)            # (N,80,2) 米制
resid = gt_m - proj_m
keep_e = 1 - (resid ** 2).sum() / (gt_m ** 2).sum()
print("\n=== H1：米制低通投影（M=24）对真值的残差 ===")
print(f"  逐点残差 |Δp| 均值 {np.linalg.norm(resid,axis=-1).mean():.5f} m"
      f"   最大 {np.linalg.norm(resid,axis=-1).max():.5f} m")
print(f"  能量保留 {keep_e:.6f}")
# 参照：如果把 160 维直接 reshape 再乘 block-diag（曾经的错误写法）
P_full = np.zeros((160, 160))
P_full[:80, :80] = P
P_full[80:, 80:] = P
flat = gt_m.reshape(len(pick), -1)
bad = flat - flat @ P_full.T
print(f"  [对照·错误写法] reshape 成 160 维 + block-diag: "
      f"残差 {np.linalg.norm(bad.reshape(-1,80,2),axis=-1).mean():.5f} m "
      f"能量保留 {1 - (bad**2).sum()/(flat**2).sum():.6f}")
# 投影后归一化空间 MSE 地板（这才是训练 loss 的地板）
resid_n = resid / sd
print(f"  → 归一化空间 MSE 地板 {float((resid_n**2).mean()):.6f}   "
      f"（实测训练 loss 卡在 0.107）")
print(f"  → 若模型完美学会投影后的真值，米制 minADE 至少 "
      f"{np.linalg.norm(resid,axis=-1).mean():.4f} m")

# ---- H2：逐元素归一化是否真的把输出变各向同性
print("\n=== H2：逐元素归一化的实际效果 ===")
print(f"  真值归一化后方差: 均值 {gt_n.var(axis=0).mean():.4f}  "
      f"p5 {np.percentile(gt_n.var(axis=0),5):.4f}  "
      f"p95 {np.percentile(gt_n.var(axis=0),95):.4f}")
# 只用全局 scale 会怎样（对照）
sc = float(nj.get("scale", 1.0))
print(f"  若只用全局 scale={sc:.3f}: 真值方差 {gt_m.var(axis=0).mean():.4f}"
      f" 范围 {gt_m.var(axis=0).min():.4f} .. {gt_m.var(axis=0).max():.4f}")
print(f"  两者方差跨度的比值: 逐元素 {gt_n.var(axis=0).max()/gt_n.var(axis=0).min():.1f}x "
      f"vs 全局 {gt_m.var(axis=0).max()/gt_m.var(axis=0).min():.1f}x")

# ---- 平凡基线：常数预测（每维取均值）在米制下多少 minADE
print(f"\n=== 平凡基线（预测训练集均值轨迹）===")
p_m = np.broadcast_to(mu.reshape(1, 80, 2), gt_m.shape)   # mu 本身已是米制
ade = np.linalg.norm(p_m - gt_m, axis=-1).mean()
print(f"  常数 mu: minADE {ade:.3f} m   "
      f"minFDE {np.linalg.norm(p_m[:,-1]-gt_m[:,-1],axis=-1).mean():.3f} m")
