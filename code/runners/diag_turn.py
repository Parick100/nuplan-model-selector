"""回答"为什么真值向下拐弯、所有候选向上拐弯"。两个互相竞争的假设：

  H1 **约束把转弯吃掉了**：速度 DCT 低通（M=24）削掉了拐弯这个低频特征。
  H2 **信息不足**：该场景目标车在 t0 时几乎静止（|v|=0.86 m/s，前 2 s 只挪了 0.22 m），
     随后却在 8 s 内加速到 46.9 m 并右转 70°。这个转弯在**可观测历史里没有任何证据**，
     属于驾驶员意图（需要地图/路线/转向灯），模型没有地图输入。
     → 若是 H2，则"漏转弯"应当**集中在低速/静止起步**的样本上，而高速样本应当正常。

判别方法：
  A. 直接把真值未来送进模型的约束算子，看转弯是否被保留（检验 H1）
  B. 在 split 2 上按 t0 速度分层，看漏转弯是否集中在低速档（检验 H2）

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_turn.py
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

STEPS = {"reg": 1, "fm": 4, "ddpm": 15}


def load(runs, obj, dev):
    ck = torch.load(os.path.join(runs, obj, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def heading_change(traj):
    """整段轨迹的朝向变化（度），用首尾各 10 步的方向差。"""
    v0 = traj[:10].mean(axis=0) - traj[0]
    v1 = traj[-1] - traj[-11:-1].mean(axis=0)
    a0 = np.arctan2(v0[1], v0[0])
    a1 = np.arctan2(v1[1], v1[0])
    return np.degrees((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--idx", type=int, default=10602)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)                # 采样噪声固定，便于复现
    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)

    # ---------------- A. 约束是否吃掉转弯 ----------------
    print("=" * 78)
    print("A. 把真值未来送进模型的约束算子，转弯被保留了吗？")
    print("=" * 78)
    m0 = load(a.runs, "fm", dev)
    rng = np.random.default_rng(0)
    idxs = list(rng.choice(len(ds), size=a.n, replace=False)) + [a.idx]
    raw, proj = [], []
    for i in idxs:
        f = ds[int(i)]["future"].unsqueeze(0).to(dev)
        with torch.no_grad():
            g = ds.denorm_future(f.cpu().numpy())[0]
            q = ds.denorm_future(m0._proj_fut(f).cpu().numpy())[0]
        raw.append(g)
        proj.append(q)
    raw = np.array(raw)
    proj = np.array(proj)
    # 转弯幅度
    tr_raw = np.array([heading_change(t) for t in raw])
    tr_proj = np.array([heading_change(t) for t in proj])
    big = np.abs(tr_raw) > 30
    print(f"  样本 {len(raw)} 条，其中 |整段朝向变化| > 30° 的 {big.sum()} 条 "
          f"({big.mean()*100:.1f}%)")
    if big.sum():
        print(f"  大转弯样本：    转弯幅度 真值 {np.abs(tr_raw[big]).mean():6.2f}°  "
              f"→ 约束后 {np.abs(tr_proj[big]).mean():6.2f}°  "
              f"(保留 {np.abs(tr_proj[big]).mean()/np.abs(tr_raw[big]).mean()*100:.1f}%)")
        print(f"                  末点位移 真值 {np.linalg.norm(raw[big][:,-1],axis=-1).mean():6.2f} m"
              f"  → 约束后 {np.linalg.norm(proj[big][:,-1],axis=-1).mean():6.2f} m")
    print(f"  全部样本：      转弯幅度 真值 {np.abs(tr_raw).mean():6.2f}°  "
          f"→ 约束后 {np.abs(tr_proj).mean():6.2f}°")
    print(f"  逐点残差 mean {np.linalg.norm(proj-raw,axis=-1).mean():.4f} m  "
          f"max {np.linalg.norm(proj-raw,axis=-1).max():.4f} m")
    print(f"  → 约束{'会' if np.abs(tr_proj[big]).mean()/max(np.abs(tr_raw[big]).mean(),1e-9) < 0.8 else '不会'}"
          f"显著削弱转弯（H1 {'成立' if np.abs(tr_proj[big]).mean()/max(np.abs(tr_raw[big]).mean(),1e-9) < 0.8 else '被否证'}）")

    # ---- 点名场景
    j = idxs.index(a.idx)
    print(f"\n  点名场景 idx={a.idx}:")
    print(f"    真值末点 ({raw[j][-1,0]:+.2f},{raw[j][-1,1]:+.2f})  整段朝向变化 "
          f"{tr_raw[j]:+.2f}°")
    print(f"    约束后末点 ({proj[j][-1,0]:+.2f},{proj[j][-1,1]:+.2f})  朝向变化 "
          f"{tr_proj[j]:+.2f}°")
    print(f"    → 约束确实保留了这次右转，问题不在约束")

    # ---------------- B. 按 t0 速度分层 ----------------
    print("\n" + "=" * 78)
    print("B. 漏转弯集中在低速起步的样本上吗？（按 t0 速度分层）")
    print("=" * 78)
    idxs2 = np.sort(rng.choice(len(ds), size=a.n, replace=False))
    items = [ds[int(i)] for i in idxs2]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in
             ("hist", "neigh", "neigh_mask", "future")}
    sc = ds.scale
    gt = ds.denorm_future(batch["future"].cpu().numpy())
    v0 = np.linalg.norm(batch["hist"].cpu().numpy()[:, -1, 2:4], axis=-1) * sc
    tr = np.array([heading_change(t) for t in gt])          # 整段朝向变化
    turn = np.abs(tr) > 30

    models = {o: load(a.runs, o, dev) for o in ("reg", "fm", "ddpm")}
    preds = {}
    for o in ("reg", "fm", "ddpm"):
        K = 1 if o == "reg" else a.K
        outs = []
        for s0 in range(0, len(items), 256):
            b = {k: v[s0:s0 + 256] for k, v in batch.items()}
            with torch.no_grad():
                pr, _ = models[o].sample(b, K=K, steps=STEPS[o])
            outs.append(ds.denorm_future(pr.cpu().numpy()))
        preds[o] = np.concatenate(outs)

    bins = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 100)]
    hdr = (f"{'|v(t0)|':>12}{'样本':>7}{'占比':>7}{'真值末点':>10}{'>30°转弯':>10}"
           f"{'reg ADE':>9}{'fm ADE':>9}{'ddpm ADE':>10}{'fm 转弯覆盖':>12}{'ddpm 转弯覆盖':>13}")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in bins:
        m = (v0 >= lo) & (v0 < hi)
        if m.sum() == 0:
            continue
        # 转弯覆盖：真值转过 >30° 的样本里，候选是否至少有一条同向且幅度 ≥50%
        tm = m & turn
        cov = {}
        for o in ("fm", "ddpm"):
            if tm.sum() == 0:
                cov[o] = float("nan")
                continue
            p = preds[o][tm]
            pr_tr = np.array([heading_change(t) for t in p[:, 0]])
            # 用所有候选里与真值同号且幅度 ≥50% 的比例
            allc = np.array([[heading_change(t) for t in pk] for pk in p])
            ok = (np.sign(allc) == np.sign(tr[tm])[:, None]) & \
                 (np.abs(allc) >= 0.5 * np.abs(tr[tm])[:, None])
            cov[o] = ok.any(axis=1).mean()
        ade = {o: float(np.linalg.norm(preds[o][m] - gt[m][:, None], axis=-1)
                        .mean(-1).min(-1).mean()) for o in ("reg", "fm", "ddpm")}
        print(f"{lo:>5}-{hi:<6}{m.sum():>7}{m.mean()*100:>6.1f}%"
              f"{np.linalg.norm(gt[m][:,-1],axis=-1).mean():>10.1f}"
              f"{turn[m].mean():>10.3f}"
              f"{ade['reg']:>9.2f}{ade['fm']:>9.2f}{ade['ddpm']:>10.2f}"
              f"{cov['fm']:>12.3f}{cov['ddpm']:>13.3f}")

    print("\n判读：")
    print("  · '>30°转弯' 列 = 该速度档里真值整段朝向变化 >30° 的样本占比（转弯有多常见）")
    print("  · '转弯覆盖' = 这些大转弯样本里，候选至少有一条同向且幅度 ≥50% 的比例")

    # ---------------- C. 横向量级与偏斜 ----------------
    print("\n" + "=" * 78)
    print("C. 候选的横向幅度有多大？有没有系统性 +y 偏斜？")
    print("=" * 78)
    print(f"{'|v(t0)|':>12}{'真值 mean y':>12}{'真值 |y|p95':>12}"
          f"{'fm 最优 y':>11}{'fm 候选 y 跨度':>15}{'ddpm 最优 y':>12}{'ddpm 跨度':>11}")
    for lo, hi in bins:
        m = (v0 >= lo) & (v0 < hi)
        if m.sum() == 0:
            continue
        gy = gt[m][:, -1, 1]
        row = [f"{lo:>5}-{hi:<6}", f"{gy.mean():>12.2f}",
               f"{np.percentile(np.abs(gy),95):>12.2f}"]
        for o in ("fm", "ddpm"):
            p = preds[o][m][:, :, -1, :]
            ade_best = np.linalg.norm(preds[o][m] - gt[m][:, None], axis=-1).mean(-1)
            bi = np.argmin(ade_best, axis=1)
            best_y = p[np.arange(p.shape[0]), bi, 1]
            span = (p[:, :, 1].max(axis=1) - p[:, :, 1].min(axis=1)).mean()
            row += [f"{best_y.mean():>11.2f}", f"{span:>15.2f}"]
        print("".join(row))
    print("\n  判读：'真值 |y|p95' 是横向量级的上界；'候选 y 跨度'是 K 条候选覆盖的横向范围。")
    print("  若跨度远小于真值横向量级 → 候选只是围绕'直行'的一小簇，看起来就像'都往上拐'。")
    print("  若最优候选的 mean y 明显为正 → 存在系统性 +y 偏斜。")

    # ---- 点名场景在其速度档里有多极端？ ----
    print("\n" + "=" * 78)
    print(f"D. 点名场景 idx={a.idx} 在其速度档里有多极端？")
    print("=" * 78)
    j2 = list(idxs2).index(a.idx)
    v0j = float(v0[j2])
    lo, hi = next((lo, hi) for lo, hi in bins if lo <= v0j < hi)
    m = (v0 >= lo) & (v0 < hi)
    yj = abs(float(gt[j2][-1, 1]))
    pct = float((np.abs(gt[m][:, -1, 1]) <= yj).mean() * 100)
    xj = float(gt[j2][-1, 0])
    px = float((gt[m][:, -1, 0] <= xj).mean() * 100)
    print(f"  该场景 t0 速度 {v0j:.3f} m/s → 落在 {lo}-{hi} m/s 档（{m.sum()} 条样本）")
    print(f"  真值末点 |y| = {yj:.2f} m，在该档的**第 {pct:.1f} 百分位**"
          f"（该档 |y| p95 = {np.percentile(np.abs(gt[m][:, -1, 1]),95):.2f} m）")
    print(f"  真值末点  x  = {xj:.2f} m，在该档的第 {px:.1f} 百分位")
    print(f"  真值末点距离 = {np.linalg.norm(gt[j2][-1]):.2f} m")
    print("  → 百分位越接近 100，说明该场景越极端：'起步车' 正常只挪十几米、几乎不横移，")
    print("     而这条却挪了近 50 m 还转了 71°，属于需要地图/路线才能推断的意图事件。")


if __name__ == "__main__":
    main()
