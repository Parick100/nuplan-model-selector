"""检查是否存在**系统性方向偏置**（可视化里所有候选都往上弯，而真值右转）。

如果模型有左右符号/偏置问题，那是一个会污染全部结果的 bug，必须在汇报前排除。
判据（不需要模型"预测得准"，只需要无偏）：

  1. 无偏性：`mean(ŷ_T − y_T)` 应≈0，且 `corr(ŷ_T, y_T) > 0`；
  2. 分条件：按真值末点 y 的符号分组（左转 / 直行 / 右转），
     各组的**平均预测 y** 必须单调跟随真值分组，而不是全部同号；
  3. 对照组：把 `neigh` 的历史 y 取反（模拟左右镜像）后重新预测，
     预测的 y 应当近似取反 —— 若不变，说明模型根本没在用横向信息。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_bias.py
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "data"))
sys.path.insert(0, os.path.join(CODE, "models"))

from dataset import MiniTrajDataset                # noqa: E402
from model import TrajectoryModel                  # noqa: E402
from train import cv_baseline                      # noqa: E402


def load(runs, obj, dev):
    ck = torch.load(os.path.join(runs, obj, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--K", type=int, default=6)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ds), size=a.n, replace=False)
    items = [ds[int(i)] for i in idx]
    logs = {ds.log_of(int(i)) for i in idx}
    print(f"split {a.split}  n={a.n}  跨 {len(logs)} 个 log")
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    gt = ds.denorm_future(batch["future"].cpu().numpy())

    # ---- 真值分组
    gy = gt[:, -1, 1]
    gx = gt[:, -1, 0]
    print(f"\n=== 真值末点分布 ===")
    print(f"  x: mean {gx.mean():7.2f}  p5 {np.percentile(gx,5):7.2f}  p95 {np.percentile(gx,95):7.2f} m")
    print(f"  y: mean {gy.mean():7.3f}  p5 {np.percentile(gy,5):7.2f}  p95 {np.percentile(gy,95):7.2f} m")
    grp = {"左转 y>+5": gy > 5.0, "直行 |y|<=5": np.abs(gy) <= 5.0, "右转 y<-5": gy < -5.0}
    for k, msk in grp.items():
        print(f"  {k:14s} {msk.sum():5d} 条  真值 mean y {gy[msk].mean():7.2f} m")

    # ---- 匀速基线（无偏参照）
    cv = cv_baseline(batch, ds.scale)[:, 0]
    print(f"\n=== 匀速外推（无偏参照）===")
    print(f"  末点 y: mean(ŷ−y) {float((cv[:,-1,1]-gy).mean()):+7.3f} m  "
          f"corr(ŷ,y) {np.corrcoef(cv[:,-1,1], gy)[0,1]:+.4f}")
    for k, msk in grp.items():
        print(f"    {k:14s} 预测 mean y {cv[msk,-1,1].mean():+7.2f}  (真值 {gy[msk].mean():+7.2f})")

    for obj in ("reg", "fm", "ddpm"):
        try:
            m, ck = load(a.runs, obj, dev)
        except Exception as e:                              # noqa: BLE001
            print(f"\n  {obj}: 载入失败 {e}")
            continue
        K = 1 if obj == "reg" else a.K
        with torch.no_grad():
            preds, nfe = m.sample(batch, K=K, steps={"reg": 1, "fm": 4, "ddpm": 15}[obj])
        p = ds.denorm_future(preds.cpu().numpy())           # (B,K,80,2)
        best = p[:, 0] if K == 1 else p[np.arange(len(gy)), np.argmin(
            np.linalg.norm(p - gt[:, None], axis=-1).mean(-1), axis=1)]
        print(f"\n=== {obj} (K={K}) ===")
        for nm, q in (("候选0", p[:, 0]), ("minADE 最优候选", best), ("候选均值", p.mean(axis=1))):
            dy = q[:, -1, 1] - gy
            dx = q[:, -1, 0] - gx
            print(f"  {nm:16s} mean(Δx) {dx.mean():+7.3f}  mean(Δy) {dy.mean():+7.3f} m  "
                  f"corr(x̂,x) {np.corrcoef(q[:,-1,0], gx)[0,1]:+.4f}  "
                  f"corr(ŷ,y) {np.corrcoef(q[:,-1,1], gy)[0,1]:+.4f}")
        print(f"  分条件（minADE 最优候选）:")
        for k, msk in grp.items():
            print(f"    {k:14s} 预测 mean y {best[msk,-1,1].mean():+7.2f}  "
                  f"(真值 {gy[msk].mean():+7.2f})   n={msk.sum()}")

        # ---- 输入消融：模型到底用了多少信息？
        # 镜像检验的比值不好解释（分子分母的分布不同），改成**直接消融**：
        # 把某一类输入置零/置均值，看 minADE 涨多少。涨幅越大 = 用得越多。
        print(f"  输入消融（minADE@K 变化，K={K}）：")
        base_ade = float(np.linalg.norm(p - gt[:, None], axis=-1).mean(-1).min(-1).mean())
        variants = {}
        v = {k: x.clone() for k, x in batch.items()}
        v["neigh"] = torch.zeros_like(v["neigh"])
        variants["邻居全部置零"] = v
        v = {k: x.clone() for k, x in batch.items()}
        v["hist"][..., 2:] = 0.0                       # 去掉目标自车速度
        variants["目标历史速度置零"] = v
        v = {k: x.clone() for k, x in batch.items()}
        v["hist"] = torch.zeros_like(v["hist"])        # 连位置一起去掉
        variants["目标历史全部置零"] = v
        v = {k: x.clone() for k, x in batch.items()}
        v["neigh"][..., 1] *= -1.0                     # 只镜像邻居的 y
        v["neigh"][..., 3] *= -1.0
        variants["仅邻居左右镜像"] = v
        print(f"    完整输入                minADE {base_ade:7.3f} m   (基准)")
        for nm, vb in variants.items():
            with torch.no_grad():
                pp, _ = m.sample(vb, K=K, steps={"reg": 1, "fm": 4, "ddpm": 15}[obj])
            q = ds.denorm_future(pp.cpu().numpy())
            ade = float(np.linalg.norm(q - gt[:, None], axis=-1).mean(-1).min(-1).mean())
            print(f"    {nm:18s} minADE {ade:7.3f} m   "
                  f"({(ade-base_ade)/max(base_ade,1e-9)*100:+7.1f}%)")

    # ---- 直行/左转/右转的覆盖率：候选里有没有出现正确方向的模式
    print("\n=== 模态覆盖（K=6 的候选方向分布）===")
    for obj in ("fm", "ddpm"):
        try:
            m, _ = load(a.runs, obj, dev)
        except Exception:                                   # noqa: BLE001
            continue
        with torch.no_grad():
            preds, _ = m.sample(batch, K=a.K, steps=4 if obj == "fm" else 15)
        p = ds.denorm_future(preds.cpu().numpy())
        py = p[:, :, -1, 1]                                  # (B,K)
        for k, msk in grp.items():
            sub = py[msk]
            if len(sub) == 0:
                continue
            # 真值在该组的方向上，有多少比例的场景"至少有一条候选落在同一侧且量级够"
            tgt = gy[msk]
            hit = (np.sign(sub) == np.sign(tgt)[:, None]) & \
                  (np.abs(sub) >= 0.5 * np.abs(tgt)[:, None])
            print(f"  {obj} {k:14s} 至少一条候选命中方向: {hit.any(axis=1).mean():.3f}  "
                  f"候选 y 范围 mean [{sub.min(axis=1).mean():+6.2f}, {sub.max(axis=1).mean():+6.2f}]"
                  f"  真值 {tgt.mean():+6.2f}")


if __name__ == "__main__":
    main()
