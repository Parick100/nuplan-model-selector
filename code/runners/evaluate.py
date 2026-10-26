"""统一评测：三个模型 × 完整指标表 + 匀速基线 + 延迟 + 多 seed。

协议 §3/§4 要求的东西一次全给：minADE/minFDE/missRate/diversity/coverage-NLL/
运动学（相对口径）/ 粗糙度倍率 / NFE / 墙钟延迟 / ≥3 seed 的均值±标准差。

默认在 **split 2**（三个模型都没见过）上评测，与选择器测试集一致。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/evaluate.py \
        --out /home/parick/dsh_ml/eval/report.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "data"))
sys.path.insert(0, os.path.join(CODE, "models"))
sys.path.insert(0, os.path.join(CODE, "metrics"))

from dataset import MiniTrajDataset                              # noqa: E402
from model import TrajectoryModel                                # noqa: E402
from traj_metrics import (coverage_nll, diversity, format_table,  # noqa: E402
                          kinematic_violation, min_ade, min_fde, miss_rate)
from train import cv_baseline                                    # noqa: E402

OBJS = ("reg", "fm", "ddpm")
LABEL = {"reg": "A 回归 (1x)", "fm": "C 流匹配 (4x)", "ddpm": "B 扩散 (15x)"}
STEPS = {"reg": 1, "fm": 4, "ddpm": 15}


def roughness(p):
    a = np.diff(p, n=2, axis=-2)
    return np.linalg.norm(a, axis=-1)


def load_model(runs, obj, dev, suffix=""):
    p = os.path.join(runs, obj + suffix, "model.pt")
    if not os.path.exists(p):
        print(f"  ! {obj}{suffix}: 无 checkpoint")
        return None, None
    ck = torch.load(p, map_location=dev)
    arch = ck.get("arch")
    if arch is None:
        print(f"  ! {obj}{suffix}: checkpoint 无 arch 字段（旧版，架构无法复现），跳过")
        return None, None
    m = TrajectoryModel(**arch).to(dev)
    try:
        m.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        print(f"  ! {obj}{suffix}: 架构与 checkpoint 不符，跳过 ({str(e).splitlines()[0][:70]})")
        return None, None
    m.eval()
    have_norm = bool(torch.any(m.sd != 1.0))
    print(f"  {obj}{suffix}: OK  code_hash={ck.get('code_hash','?')}  "
          f"steps={ck.get('args',{}).get('steps','?')}  "
          f"basis_m={arch.get('basis_m','?')}  loss_power={arch.get('loss_power','?')}  "
          f"wta_k={arch.get('wta_k','?')}  "
          f"归一化buffer={'有' if have_norm else '!! 缺失（mu/sd 退回恒等）'}")
    return m, ck


def make_batch(ds, n, batch_size, dev, seed=1234):
    """跨 log 的固定种子随机子集（协议要求，避免"单 log 连续帧"当验证集）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    idx = np.sort(idx)
    logs = sorted({ds.log_of(int(i)) for i in idx})
    bs = []
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        items = [ds[int(i)] for i in chunk]
        bs.append({k: torch.stack([it[k] for it in items]).to(dev)
                   for k in ("hist", "neigh", "neigh_mask", "future")})
    print(f"评测集: {len(idx)} 样本, 跨 {len(logs)} 个 log, {len(bs)} 个 batch")
    return bs, idx, logs


@torch.no_grad()
def run_model(m, bs, ds, K, steps, dev, seed):
    """返回拼接后的预测 (N,K,80,2) 米制 与真值 (N,80,2) 米制。"""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    P, G = [], []
    for b in bs:
        preds, _ = m.sample(b, K=K, steps=steps, generator=g)
        P.append(ds.denorm_future(preds.cpu().numpy()))
        G.append(ds.denorm_future(b["future"].cpu().numpy()))
    return np.concatenate(P), np.concatenate(G)


@torch.no_grad()
def latency(m, ds, dev, K, steps, batch, reps=20):
    """墙钟延迟（毫秒/次预测）。batch=1 与 batch=64 各报一次（协议 §4）。"""
    items = [ds[i] for i in range(batch)]
    b = {k: torch.stack([it[k] for it in items]).to(dev)
         for k in ("hist", "neigh", "neigh_mask", "future")}
    for _ in range(3):                                   # 预热
        m.sample(b, K=K, steps=steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        m.sample(b, K=K, steps=steps)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def metrics_for(p, g, K):
    """单组预测的完整指标。p (N,K,T,2)，g (N,T,2) 米制。"""
    ks = kinematic_violation(p, g)
    return {
        "K": K,
        "minADE": float(min_ade(p, g, K).mean()),
        "minFDE": float(min_fde(p, g, K).mean()),
        "missRate": float(miss_rate(p, g, K)),
        "diversity": float(diversity(p, K).mean()),
        "coverage_nll": float(coverage_nll(p, g, K).mean()),
        "rough_ratio": float(roughness(p).mean() / max(roughness(g).mean(), 1e-9)),
        "rough_gt": float(roughness(g).mean()),
        "accel_violation": ks["accel_violation"],
        "jerk_violation": ks["jerk_violation"],
        "accel_thr": ks["accel_thr"],
        "jerk_thr": ks["jerk_thr"],
        "kin_note": ks["note"],
        "accel_p95_ratio": ks["accel_p95_ratio"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--objs", default="reg,fm,ddpm")
    ap.add_argument("--suffix", default="",
                    help="checkpoint 目录后缀，如 _lp2 → 读 runs/reg_lp2/model.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    objs = [o for o in a.objs.split(",") if o]

    print(f"device={dev}  split={a.split}  n={a.n}  K={a.K}  seeds={a.seeds}  "
          f"suffix='{a.suffix}'")
    print("=== 载入模型 ===")
    models = {}
    for o in objs:
        models[o], _ = load_model(a.runs, o, dev, a.suffix)

    ds = MiniTrajDataset(a.root, split=a.split, cache_logs=2)
    bs, idx, logs = make_batch(ds, a.n, a.batch_size, dev)

    # ---- 参考基线 ----
    print("\n=== 基线 ===")
    G = np.concatenate([ds.denorm_future(b["future"].cpu().numpy()) for b in bs])
    CV = np.concatenate([cv_baseline(b, ds.scale) for b in bs])
    base = {"K=1": metrics_for(CV, G, 1)}
    # 真值自身在相对口径下的运动学基准（**应当** ≈0.05，而不是 1.000）
    gt_kin = kinematic_violation(G[:, None], G)
    base["gt_kinematic"] = gt_kin
    print(f"  匀速外推        minADE {base['K=1']['minADE']:7.3f}  "
          f"minFDE {base['K=1']['minFDE']:7.3f}  miss {base['K=1']['missRate']:.3f}  "
          f"粗糙度 {base['K=1']['rough_ratio']:6.2f}x")
    print(f"  真值            |d²p| {base['K=1']['rough_gt']:.4f} m")
    print(f"  真值运动学基准  |a| p50/p95 {gt_kin['accel_p50']:.2f}/{gt_kin['accel_p95']:.2f} m/s²"
          f"  逐点超限率 {gt_kin['accel_violation']:.3f}"
          f"  ← 相对口径下必须 ≈0.05（旧实现是 1.000）")
    print(f"                  阈值 |a|>{gt_kin['accel_thr']:.2f} m/s²  "
          f"|j|>{gt_kin['jerk_thr']:.1f} m/s³")

    # ---- 各模型 ----
    report = {"split": a.split, "n": len(idx), "n_logs": len(logs), "logs": logs,
              "K": a.K, "seeds": a.seeds, "baseline": base, "models": {}}
    for o in objs:
        m = models[o]
        if m is None:
            continue
        # 回归模型天然只有 1 条候选（K 无意义），如实报 K=1 而不是硬凑成 K=6
        K = 1 if o == "reg" else a.K
        print(f"\n=== {LABEL.get(o, o)}  (K={K}, steps={STEPS[o]}) ===")
        per_seed = []
        for s in range(a.seeds):
            seed = 1000 + s
            p, g = run_model(m, bs, ds, K, STEPS[o], dev, seed)
            mm = metrics_for(p, g, K)
            mm["seed"] = seed
            per_seed.append(mm)
            if K > 1:                       # 再报一份 K=1，便于和回归公平对比
                per_seed.append(dict(metrics_for(p[:, :1], g, 1), seed=seed))
            print(f"  seed {seed}: minADE {mm['minADE']:7.3f}  minFDE {mm['minFDE']:7.3f}  "
                  f"miss {mm['missRate']:.3f}  div {mm['diversity']:6.2f}  "
                  f"粗糙度 {mm['rough_ratio']:6.2f}x  |a|p95/GT {mm['accel_p95_ratio']:6.2f}")

        # 汇总：只对该模型的主 K 取 seed 统计，避免把 K=1 混进来
        main_seeds = [x for x in per_seed if x["K"] == K]
        agg = {}
        for key in ("minADE", "minFDE", "missRate", "diversity", "coverage_nll",
                    "rough_ratio", "accel_p95_ratio"):
            v = np.array([x[key] for x in main_seeds])
            agg[key] = {"mean": float(v.mean()),
                        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        lat1 = latency(m, ds, dev, K, STEPS[o], 1)
        lat64 = latency(m, ds, dev, K, STEPS[o], 64)
        nfe = K * STEPS[o]
        report["models"][o] = {"label": LABEL.get(o, o), "K": K, "per_seed": per_seed,
                               "agg": agg, "nfe": nfe,
                               "latency_ms_b1": lat1, "latency_ms_b64": lat64,
                               "params": m.param_report()}
        print(f"  → 汇总 minADE {agg['minADE']['mean']:.3f}±{agg['minADE']['std']:.3f}  "
              f"minFDE {agg['minFDE']['mean']:.3f}±{agg['minFDE']['std']:.3f}")
        print(f"  → NFE {nfe}  延迟 batch1 {lat1:.1f} ms  batch64 {lat64:.1f} ms  "
              f"(单次预测 {lat64:.1f} ms 处理 {64} 条)")

    # ---- 总结表 ----
    print("\n\n" + "=" * 78)
    print(f"总结（split {a.split}，{len(idx)} 样本 × {a.seeds} seed）")
    print("=" * 78)
    hdr = (f"{'模型':<18}{'NFE':>5}{'minADE':>16}{'minFDE':>16}{'miss':>8}"
           f"{'粗糙度':>9}{'延迟ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    cv = base["K=1"]
    print(f"{'匀速外推(基线)':<16}{1:>5}{cv['minADE']:>16.3f}{cv['minFDE']:>16.3f}"
          f"{cv['missRate']:>8.3f}{cv['rough_ratio']:>9.2f}{'—':>9}")
    for o in objs:
        if o not in report["models"]:
            continue
        R = report["models"][o]
        ag = R["agg"]
        print(f"{R['label']:<16}{R['nfe']:>5}"
              f"{ag['minADE']['mean']:>10.3f}±{ag['minADE']['std']:<4.2f}"
              f"{ag['minFDE']['mean']:>10.3f}±{ag['minFDE']['std']:<4.2f}"
              f"{ag['missRate']['mean']:>8.3f}"
              f"{ag['rough_ratio']['mean']:>9.2f}{R['latency_ms_b1']:>9.1f}")
    print("\n（粗糙度 = |d²p|预测 / |d²p|真值；1.0 表示与真值同样平滑）")
    print(f"（运动学为逐点相对口径；真值基准超限率 "
          f"{gt_kin['accel_violation']:.3f} == p95 设计值 0.05，|a|p95/真值 真值自己 = 1.00）")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n报告已写入 {a.out}")


if __name__ == "__main__":
    main()
