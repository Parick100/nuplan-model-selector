"""修正版转弯覆盖统计：先前那张表的高速档只用了 13 条样本，不能报。

做法：
  1. 从**整个 split 2**（61,026 条）统计各速度档的真实占比与横移分位（不需要模型，很快）；
  2. 对"需要模型"的覆盖统计，在每个档里**按需平衡抽样**（每档最多 800 条大转弯样本），
     这样每格的 n 都够大，并**把 n 一并打印**。

⚠ 教训：`diag_turn.py` 的 B 表里 "10-100 m/s" 档的转弯覆盖是 1/13 —— 一个 n=13 的
比例不能当结论。覆盖率必须和样本数一起报。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_coverage.py
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
from diag_turn import heading_change               # noqa: E402

STEPS = {"reg": 1, "fm": 4, "ddpm": 15}
BINS = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 100)]


def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def sample_T(m, batch, K, steps, temp, dev, seed=0):
    from model import alpha_bar
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
    ap.add_argument("--per-bin", type=int, default=800)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--suffix", default="", help="checkpoint 目录后缀，如 _lp2")
    ap.add_argument("--temps", default="1.0,2.0")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    sc = ds.scale

    # ---------- 1) 全 split 统计（不需要模型） ----------
    # ⚠ `iter_logs()` 交回的是**原始数组（已经是米制）**；归一化发生在
    #   MiniTrajDataset.__getitem__ 里（hist/scale、future→(f−mu)/sd）。
    #   这里第一版又把 hist 乘了一次 scale、把 future 又反归一化了一次，
    #   于是速度虚高 20.6 倍、末点距离算到 400 m，分档全错。
    #   教训：走 __getitem__ 的路径与走 mmap 原始数组的路径单位不同，不能混用。
    print(f"=== 整个 split {a.split}（{len(ds):,} 条）的速度分层与转弯占比 ===")
    V, TURN, YT, XT = [], [], [], []
    for e, arr in ds.iter_logs():
        n = arr["hist"].shape[0]
        if n == 0:
            continue
        h = np.asarray(arr["hist"])[:, -1, 2:4].astype(np.float64)   # 原值即 m/s
        f = np.asarray(arr["future"]).astype(np.float64)             # 原值即米
        V.append(np.linalg.norm(h, axis=-1))
        TURN.append(np.array([heading_change(t) for t in f]))
        YT.append(f[:, -1, 1])
        XT.append(f[:, -1, 0])
    V = np.concatenate(V)
    TURN = np.concatenate(TURN)
    YT = np.concatenate(YT)
    XT = np.concatenate(XT)
    turn = np.abs(TURN) > 30.0
    print(f"{'|v(t0)|':>12}{'样本':>9}{'占比':>8}{'>30°转弯':>10}{'转弯样本数':>11}"
          f"{'|y_T| p50':>11}{'|y_T| p95':>11}{'|p_T| p50':>11}")
    for lo, hi in BINS:
        m = (V >= lo) & (V < hi)
        if m.sum() == 0:
            continue
        print(f"{lo:>5}-{hi:<6}{m.sum():>9}{m.mean()*100:>7.1f}%{turn[m].mean():>10.3f}"
              f"{int(turn[m].sum()):>11}"
              f"{np.median(np.abs(YT[m])):>11.2f}"
              f"{np.percentile(np.abs(YT[m]),95):>11.2f}"
              f"{np.median(np.linalg.norm(np.stack([XT[m],YT[m]],-1),axis=-1)):>11.1f}")
    m0 = V >= 100
    if m0.any():
        print(f"  ⚠ |v(t0)|>=100 m/s 的样本 {int(m0.sum())} 条（{m0.mean()*100:.1f}%）——"
              f"若不为 0，说明单位又搞错了（车辆不可能 >=360 km/h）")
    else:
        print(f"  ✓ 没有 |v(t0)|>=100 m/s 的样本（分档覆盖了全部 {len(V):,} 条）")

    # ---------- 2) 每档平衡抽样，估覆盖 ----------
    print(f"\n=== 转弯覆盖（每档最多 {a.per_bin} 条 >30° 转弯样本，n 一并报出）===")
    rng = np.random.default_rng(0)
    sel = []
    for lo, hi in BINS:
        m = (V >= lo) & (V < hi) & turn
        ids = np.nonzero(m)[0]
        if len(ids) == 0:
            continue
        ids = ids if len(ids) <= a.per_bin else rng.choice(ids, a.per_bin, replace=False)
        sel.append((lo, hi, ids))
    print(f"  实际送入模型的样本: {sum(len(i) for _, _, i in sel)} 条")

    items, meta = [], []
    for lo, hi, ids in sel:
        for i in ids:
            items.append(ds[int(i)])
            meta.append((lo, hi))
    meta = np.array([f"{lo}-{hi}" for lo, hi in meta])
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    tr = np.array([heading_change(t) for t in gt])

    for obj in ("fm", "ddpm"):
        m = load(a.runs, obj, dev, a.suffix)
        for temp in [float(x) for x in a.temps.split(",")]:
            outs = []
            for s0 in range(0, len(items), 256):
                b = {k: v[s0:s0 + 256] for k, v in batch.items()}
                outs.append(ds.denorm_future(
                    sample_T(m, b, a.K, STEPS[obj], temp, dev).cpu().numpy()))
            p = np.concatenate(outs)
            allc = np.array([[heading_change(t) for t in pk] for pk in p])
            ok = (np.sign(allc) == np.sign(tr)[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr)[:, None])
            hit = ok.any(axis=1)
            line = f"  {obj:5s} T={temp:.1f}  总体 {hit.mean():.3f} (n={len(hit)})  "
            for lo, hi in BINS:
                mm = meta == f"{lo}-{hi}"
                if mm.sum():
                    line += f"| {lo}-{hi}: {hit[mm].mean():.3f}(n={int(mm.sum())}) "
            ade = np.linalg.norm(p - gt[:, None], axis=-1).mean(-1).min(-1).mean()
            print(f"{line}\n        minADE@K {ade:.3f}")
        del m

    print("\n判读：只有 n 够大的格子才可信。若某档 n < 100，不要把它的比例当结论。")
    print("温度 T>1 拉开候选 → 覆盖上升但 minADE 可能变差，属于精度/覆盖的取舍，")
    print("必须三条臂同口径重跑评测后才可作为正式结果。")


if __name__ == "__main__":
    main()
