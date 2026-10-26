"""诊断：流匹配 / 扩散的**场**误差随 t 的分布，以及 t=1 处是否在照抄输入噪声。

两种范式现在都预测 x0，所以正确的行为是：
  · 场误差在整个 t 上**大致平坦**（这正是放弃 ε/v 参数化的目的）；
  · t=1 时 x_t 是纯噪声，与 x0 独立，因此 **corr(x̂0, x_t) ≈ 0**。
    若接近 1，说明模型在照抄输入而不是用上下文预测 —— 那样采样就甩不掉初始噪声。
  · t→0 时 x̂0 应≈ x_t（此时 x_t 几乎就是 x0）。

⚠ 审查指出的旧 bug：本脚本原先把 `[ds[i] for i in range(n)]` 当验证集，
而全局下标 0..n-1 **全部来自同一个 log**（split 1 的前 64/512 条都属于
`2021.05.12.22.00.38`）—— 那个"单 log 验证集"缺陷在 train.py 里修好了，
却在诊断脚本里活了下来，于是逐 t 曲线仍带 log 偏置。现在统一用跨 log 随机子集。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_fm.py --objective fm
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
sys.path.insert(0, os.path.join(CODE, "metrics"))

from dataset import MiniTrajDataset                # noqa: E402
from model import TrajectoryModel                  # noqa: E402
from traj_metrics import min_ade, min_fde          # noqa: E402


def wiggle(t):                                     # (..., T, 2) -> 每步二阶差分模长
    return np.linalg.norm(np.diff(t, n=2, axis=-2), axis=-1).mean()


def cross_log_batch(ds, n, seed=1234):
    """跨 log 的固定种子随机子集（与 train.py 的验证集口径一致）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    logs = sorted({ds.log_of(int(i)) for i in idx})
    print(f"诊断集: {len(idx)} 样本, 跨 {len(logs)} 个 log")
    if len(logs) <= 1:
        sys.exit(f"!! 只覆盖了 {len(logs)} 个 log，诊断会有 log 偏置")
    items = [ds[int(i)] for i in idx]
    return {k: torch.stack([it[k] for it in items]) for k in
            ("hist", "neigh", "neigh_mask", "future")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--objective", default="fm", choices=["fm", "ddpm"])
    ap.add_argument("--ckpt", default=None, help="直接指定 checkpoint 路径")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--split", type=int, default=1)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = a.ckpt or os.path.join(a.runs, a.objective, "model.pt")
    ck = torch.load(ckpt, map_location=dev)
    print(f"ckpt: {ckpt}")
    arch = ck.get("arch")
    if arch is None:
        sys.exit("!! checkpoint 无 arch 字段，无法复现架构")
    m = TrajectoryModel(**arch).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if not bool(torch.any(m.sd != 1.0)):
        sys.exit("!! mu/sd buffer 未恢复，归一化退回恒等")

    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    batch = cross_log_batch(ds, a.n)
    batch = {k: v.to(dev) for k, v in batch.items()}
    gt = ds.denorm_future(batch["future"].cpu().numpy())

    print(f"{a.objective}  n={a.n}  K={a.K}   basis_m={arch.get('basis_m','?')}  "
          f"dof={m.output_dim}")
    print(f"真值 粗糙度 |d²p| = {wiggle(gt):.4f} m\n")
    print(f"{'steps':>6} {'nfe':>6} {'minADE':>9} {'minFDE':>9} {'粗糙度':>10} {'vs GT':>8}")
    for steps in (1, 2, 4, 8, 16, 32, 64):
        torch.manual_seed(0)
        preds, nfe = m.sample(batch, K=a.K, steps=steps)
        p = ds.denorm_future(preds.cpu().numpy())
        sw = wiggle(p)
        print(f"{steps:>6} {nfe:>6} {min_ade(p, gt).mean():>9.3f} "
              f"{min_fde(p, gt).mean():>9.3f} {sw:>10.4f} {sw/wiggle(gt):>7.2f}x")

    # ---- t=1 / t=0 处的场行为
    print("\n=== 场诊断（两种范式都以 x0 为目标）===")
    with torch.no_grad():
        tokens, mask, _ = m.encode(batch)
        B = tokens.shape[0]
        for tval in (1.0, 0.5, 0.0):
            torch.manual_seed(0)
            xt = torch.randn(B, m.traj_len, 2, device=dev)
            x0n = batch["future"]
            if tval == 0.0:
                xt = x0n.clone()
            x_in = torch.cat([torch.zeros(B, 1, 2, device=dev), xt], dim=1).reshape(B, -1)
            o = m.predict_field(x_in, torch.full((B,), tval, device=dev),
                                tokens, mask).reshape(B, m.out_len, 2)[:, 1:]
            cc = float(torch.corrcoef(torch.stack([o.reshape(-1), xt.reshape(-1)]))[0, 1])
            # 米制误：把 x̂0 与（若 t=0 则真值，否则与真值无关）比较
            o_m = ds.denorm_future(o.cpu().numpy())
            err = float(np.linalg.norm(o_m - gt, axis=-1).mean())
            print(f"  t={tval:.1f}: ‖x̂0‖ {float(o.norm()/(B**.5)):9.3f}  "
                  f"corr(x̂0, x_t) {cc:+.4f}   米制 |x̂0 − 真值| {err:8.3f} m")
        print("  判读：")
        print("    t=1 的 corr 应 ≈0（x_t 是纯噪声，与 x0 独立，模型必须靠上下文）")
        print("    t=0 的 corr 应 ≈1（此时 x_t 就是 x0，场应恒等），"
              "米制误差应 ≈ 约束地板")


if __name__ == "__main__":
    main()
