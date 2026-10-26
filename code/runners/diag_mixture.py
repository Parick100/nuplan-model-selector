"""推理期候选混合：能否同时拿到 T=1 的精度和 T=2 的覆盖？

背景（实测）：
  · 温度 T>1 会把候选拉开，转弯覆盖大涨，但在**全量** split 2 上 minADE 变差；
  · T=1 精度最好但候选太窄（横向跨度 2.9–9.0 m，真值横移 p95 9.5–33 m）。

于是试一个免费的推理期做法：**K 条候选不取自同一个温度**，
一部分用 T=1（保精度），一部分用 T>1（保覆盖）。取 min 的操作会自动挑选最合适的那条，
所以 minADE@K 只会变好或持平。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_mixture.py --suffix _lp2
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
from diag_coverage import sample_T                 # noqa: E402
from diag_turn import heading_change               # noqa: E402

STEPS = {"reg": 1, "fm": 4, "ddpm": 15}


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    rng = np.random.default_rng(1234)
    idx = np.sort(rng.choice(len(ds), size=a.n, replace=False))
    items = [ds[int(i)] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    tr = np.array([heading_change(t) for t in gt])
    turn = np.abs(tr) > 30
    print(f"split {a.split}  n={len(idx)}  K={a.K}  大转弯样本 {turn.sum()}")
    print("注：'混合' 行的 minADE 是 K 条混合候选上取 min 的结果，")
    print("    理论上下界就是各单温度方案里更好的那条；混合的价值在覆盖与稳健性。\n")

    # 方案：(名称, [每条候选的温度])；长度必须 == K
    K = a.K
    schemes = [
        ("全 T=1.0", [1.0] * K),
        ("全 T=1.5", [1.5] * K),
        ("全 T=2.0", [2.0] * K),
        ("4x1.0 + 2x1.5", [1.0] * 4 + [1.5] * 2),
        ("4x1.0 + 2x2.0", [1.0] * 4 + [2.0] * 2),
        ("3x1.0 + 3x1.5", [1.0] * 3 + [1.5] * 3),
        ("2x1.0+2x1.5+2x2.0", [1.0, 1.0, 1.5, 1.5, 2.0, 2.0]),
    ]
    for obj in ("fm", "ddpm"):
        m = load(a.runs, obj, dev, a.suffix)
        print(f"=== {obj} ({STEPS[obj]} 步, K={K}) ===")
        hdr = (f"{'方案':>20}{'minADE@K':>10}{'minFDE@K':>10}{'miss':>8}"
               f"{'转弯覆盖':>10}{'高速转弯覆盖':>14}")
        print(hdr)
        print("-" * len(hdr))
        fast = turn & (np.linalg.norm(batch["hist"].cpu().numpy()[:, -1, 2:4],
                                     axis=-1) * ds.scale >= 10.0)
        for name, temps in schemes:
            outs = []
            for c, T in enumerate(temps):
                ch = []
                for s0 in range(0, len(items), 256):
                    b = {k: v[s0:s0 + 256] for k, v in batch.items()}
                    ch.append(ds.denorm_future(
                        sample_T(m, b, 1, STEPS[obj], T, dev, seed=1000 + c).cpu().numpy()))
                outs.append(np.concatenate(ch))          # (N,1,80,2)
            p = np.concatenate(outs, axis=1)             # (N,K,80,2)
            ade = np.linalg.norm(p - gt[:, None], axis=-1).mean(-1).min(-1).mean()
            fde = np.linalg.norm(p[:, :, -1] - gt[:, None, -1], axis=-1).min(-1).mean()
            miss = (np.linalg.norm(p[:, :, -1] - gt[:, None, -1],
                                   axis=-1).min(-1) > 2.0).mean()
            allc = np.array([[heading_change(t) for t in pk] for pk in p[turn]])
            ok = (np.sign(allc) == np.sign(tr[turn])[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr[turn])[:, None])
            cov = ok.any(axis=1).mean()
            if fast.sum():
                af = np.array([[heading_change(t) for t in pk] for pk in p[fast]])
                okf = (np.sign(af) == np.sign(tr[fast])[:, None]) & \
                      (np.abs(af) >= 0.5 * np.abs(tr[fast])[:, None])
                covf = okf.any(axis=1).mean()
            else:
                covf = float("nan")
            print(f"{name:>20}{ade:>10.3f}{fde:>10.3f}{miss:>8.3f}"
                  f"{cov:>10.3f}{covf:>14.3f}(n={int(fast.sum())})")
        print()
        del m

    print("判读：若某个混合方案同时拿到最低 minADE 和较高覆盖 → 直接采用，零训练成本。")
    print("      若覆盖涨但 minADE 也涨 → 是取舍，必须作为显式工作点报告。")


if __name__ == "__main__":
    main()
