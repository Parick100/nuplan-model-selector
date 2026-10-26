"""诊断结论的最后一个验证：候选铺得太窄，是"目标函数收缩"还是"采样噪声不够"？

如果原因是**采样分布太窄**（MSE 目标把 x0 往条件均值收缩），那么把初始噪声
放大 T>1 倍应当直接拉开候选间距，并提升 minADE@6 / 转弯覆盖。
若加大 T 完全没用，说明收缩发生在模型权重里，只能改目标函数。

⚠ 有意**不改 model.py**：采样循环在这里独立重写一份，这样三条已训 checkpoint 的
`code_hash` 与文件仍然一一对应，审计链不断。温度只是一个**推理期操作点**，
若要采用必须三条臂同口径重跑评测。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_temperature.py
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
from model import TrajectoryModel, alpha_bar       # noqa: E402
from diag_turn import heading_change               # noqa: E402


def load(runs, obj, dev):
    ck = torch.load(os.path.join(runs, obj, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def sample_T(m, batch, K, steps, temp, dev, seed=0):
    """与 model.sample 相同的 DDIM 循环，只把初始噪声乘上 temp。"""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    tokens, mask, _ = m.encode(batch)
    B = tokens.shape[0]
    C = B * K
    tok = tokens.repeat_interleave(K, dim=0)
    msk = mask.repeat_interleave(K, dim=0)
    x = torch.randn(C, m.out_len * 2, device=dev, generator=g) * temp
    x = x.reshape(C, m.out_len, 2)
    x[:, 0, :] = 0.0
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        out = m.predict_field(x.reshape(C, -1), t.expand(C), tok, msk)
        out = out.reshape(C, m.out_len, 2)
        out[:, 0, :] = 0.0
        if m.objective == "ddpm":
            ab, abn = alpha_bar(t), alpha_bar(tn)
            eps = (x - torch.sqrt(ab) * out) / torch.sqrt(1 - ab)
            x = torch.sqrt(abn) * out + torch.sqrt(1 - abn) * eps
        else:
            eps = (x - (1 - t) * out) / t
            x = (1 - tn) * out + tn * eps
        x[:, 0, :] = 0.0
    return x[:, 1:, :].reshape(B, K, m.traj_len, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--K", type=int, default=6)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    rng = np.random.default_rng(0)
    idxs = np.sort(rng.choice(len(ds), size=a.n, replace=False))
    items = [ds[int(i)] for i in idxs]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    sc = ds.scale
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    tr = np.array([heading_change(t) for t in gt])
    turn = np.abs(tr) > 30
    v0 = np.linalg.norm(batch["hist"].cpu().numpy()[:, -1, 2:4], axis=-1) * sc
    fast = v0 >= 10.0
    rough_gt = np.linalg.norm(np.diff(gt, n=2, axis=-2), axis=-1).mean()

    print(f"split {a.split}  n={a.n}  K={a.K}   大转弯样本(>30°) {turn.sum()}，"
          f"其中高速档(>=10 m/s) {int((turn & fast).sum())}")

    for obj, steps in (("fm", 4), ("ddpm", 15)):
        m = load(a.runs, obj, dev)
        print(f"\n=== {obj} ({steps} 步) ===")
        hdr = (f"{'T':>5}{'minADE@K':>10}{'minFDE@K':>10}{'diversity':>11}"
               f"{'粗糙度':>9}{'转弯覆盖':>10}{'高速转弯覆盖':>14}")
        print(hdr)
        print("-" * len(hdr))
        for temp in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            outs = []
            for s0 in range(0, len(items), 256):
                b = {k: v[s0:s0 + 256] for k, v in batch.items()}
                outs.append(ds.denorm_future(
                    sample_T(m, b, a.K, steps, temp, dev).cpu().numpy()))
            p = np.concatenate(outs)                        # (N,K,80,2)
            ade = np.linalg.norm(p - gt[:, None], axis=-1).mean(-1).min(-1).mean()
            fde = np.linalg.norm(p[:, :, -1] - gt[:, None, -1], axis=-1).min(-1).mean()
            ends = p[:, :, -1, :]
            div = np.linalg.norm(ends[:, :, None] - ends[:, None, :], axis=-1)
            div = div[np.triu_indices(a.K, 1)].mean() if a.K > 1 else 0.0
            rgh = np.linalg.norm(np.diff(p, n=2, axis=-2), axis=-1).mean() / rough_gt
            allc = np.array([[heading_change(t) for t in pk] for pk in p[turn]])
            ok = (np.sign(allc) == np.sign(tr[turn])[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr[turn])[:, None])
            cov = ok.any(axis=1).mean()
            tf = turn & fast
            allf = np.array([[heading_change(t) for t in pk] for pk in p[tf]])
            okf = (np.sign(allf) == np.sign(tr[tf])[:, None]) & \
                  (np.abs(allf) >= 0.5 * np.abs(tr[tf])[:, None])
            covf = okf.any(axis=1).mean() if tf.sum() else float("nan")
            print(f"{temp:>5.1f}{ade:>10.3f}{fde:>10.3f}{div:>11.2f}"
                  f"{rgh:>9.2f}{cov:>10.3f}{covf:>14.3f}")
        del m

    print("\n判读：")
    print("  · 若 T>1 明显提升 minADE@K 与转弯覆盖 → 诊断成立：候选被 MSE 收缩得太窄，")
    print("    加大采样噪声是**推理期的现成旋钮**（但要三条臂同口径重跑评测才算数）。")
    print("  · 若 T 增大只让 minADE 变差（候选跑飞）→ 收缩在权重里，必须改目标函数。")


if __name__ == "__main__":
    main()
