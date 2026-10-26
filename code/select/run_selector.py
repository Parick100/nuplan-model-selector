"""模型选择器：把"三臂对比"变成"给定算力预算该跑哪个模型"。

协议 §6 的落地。整体流程：

  1. **打标**（split 1，模型没在它上面训练过）：对每个场景跑三条臂，得到
     q(i,m) = minADE@K，以及代价 c(m) = NFE。
  2. **go / no-go 关卡**（§6.0）：先算 oracle 差距
         gap = mean_i min_m q(i,m)   −   min_m mean_i q(i,m)
     即"每场都选最优"比"固定用最好的单模型"好多少。若 gap ≈ 0，
     选择器在数学上不可能有帮助，必须如实报告并停止 —— 这是本脚本最重要的一行输出。
  3. **特征**（§6.2）：只用推理时确实拿得到的东西 —— 目标车历史统计量、邻居统计量、
     scenario_tag，以及**回归臂自己的预测**（它只要 1 次前向，先跑它是免费的）。
  4. **级联二分类门**（§6.3）：代价有序 1× < 4× < 15×，所以用两个门而不是三分类：
         gate A: 回归够好吗？  → 用回归
         gate B: 流匹配够好吗？→ 用流匹配
         否则                 → 用扩散
     标签由阈值 τ 定义：m*(i) = argmin_m c(m) s.t. q(i,m) ≤ τ。
  5. **按策略评估**（§6.4）：在 **split 2** 上评"最终精度 / 平均延迟 / 满足质量要求的比例"，
     而不是评分类器准确率；并与三条固定臂、oracle、以及线性插值基线比较。

    python code/select/run_selector.py --label-split 1 --label-n 12000 \
        --test-split 2 --test-n 4096 --suffix _lp2
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
sys.path.insert(0, os.path.join(CODE, "runners"))

from dataset import MiniTrajDataset                # noqa: E402
from model import TrajectoryModel                  # noqa: E402
from train import cv_baseline                      # noqa: E402

OBJS = ("reg", "fm", "ddpm")
STEPS = {"reg": 1, "fm": 4, "ddpm": 15}
NFE = {"reg": 1, "fm": 24, "ddpm": 90}             # K=6
LATENCY_MS = {"reg": 6.2, "fm": 18.9, "ddpm": 59.5}  # batch-1, measured by evaluate.py


# --------------------------------------------------------------- 模型
def load(runs, obj, dev, suffix=""):
    ck = torch.load(os.path.join(runs, obj + suffix, "model.pt"), map_location=dev)
    m = TrajectoryModel(**ck["arch"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


# --------------------------------------------------------------- 特征
def ang(v):
    return np.arctan2(v[..., 1], v[..., 0])


def build_features(batch, ds, reg_pred):
    """全部特征都是**推理时可得**的，且不需要额外前向（回归臂本来就要跑）。

    batch: 已归一化张量；reg_pred: 回归臂的预测 (N,K,80,2) 米制。
    """
    sc = ds.scale
    hist = batch["hist"].cpu().numpy() * sc                      # (N,20,4) 米
    neigh = batch["neigh"].cpu().numpy() * sc                    # (N,10,20,6)
    nmask = batch["neigh_mask"].cpu().numpy().astype(bool)       # (N,10)
    tags = batch.get("tags")
    tags = tags.cpu().numpy() if tags is not None else None
    N = hist.shape[0]

    v = hist[:, :, 2:4]                                          # (N,20,2)
    sp = np.linalg.norm(v, axis=-1)                              # (N,20)
    pos = hist[:, :, :2]
    d = np.diff(pos, axis=1)
    path = np.linalg.norm(d, axis=-1).sum(axis=1)
    # 朝向变化率：最后两段速度方向的夹角
    a1 = ang(v[:, -1])
    a2 = ang(v[:, -3])
    yaw_rate = np.abs((a1 - a2 + np.pi) % (2 * np.pi) - np.pi)
    # 曲率代理：横向位移 / 路径长度
    straight = np.linalg.norm(pos[:, -1] - pos[:, 0], axis=-1)
    curv = 1.0 - straight / np.maximum(path, 1e-3)
    acc = (sp[:, -1] - sp[:, -3]) / 0.2

    # 邻居：有效数、最近距离、平均速度
    nvalid = nmask.sum(axis=1).astype(np.float32)
    dist = np.linalg.norm(neigh[:, :, -1, :2], axis=-1)          # (N,10)
    dist = np.where(nmask, dist, np.inf)
    dmin = dist.min(axis=1)                                      # (N,) 无邻居时是 inf
    nearest = np.where(np.isfinite(dmin), dmin, 100.0)
    nsp = np.where(nmask, np.linalg.norm(neigh[:, :, -1, 2:4], axis=-1), 0.0)
    mean_nsp = nsp.sum(axis=1) / np.maximum(nvalid, 1)

    # 回归臂自己的预测（免费，1 NFE）
    rp = reg_pred[:, 0]                                          # (N,80,2)
    reg_end = np.linalg.norm(rp[:, -1], axis=-1)
    reg_curv = np.linalg.norm(np.diff(rp, n=2, axis=1), axis=-1).mean(axis=1)
    reg_end_y = rp[:, -1, 1]
    cv = cv_baseline(batch, sc)[:, 0]
    disagree = np.linalg.norm(rp[:, -1] - cv[:, -1], axis=-1)
    # 常速度 vs 常加速度外推的分歧：多模态/机动风险的便宜代理
    v_last = hist[:, -1, 2:4]
    a_last = (hist[:, -1, 2:4] - hist[:, -3, 2:4]) / 0.2
    t = (np.arange(1, 81) * 0.1)[None, :, None]
    ca_end = v_last[:, None, :] * t[:, :, 0:1] + 0.5 * a_last[:, None, :] * t ** 2
    cvca = np.linalg.norm(ca_end[:, -1] - cv[:, -1], axis=-1)

    feats = [sp[:, -1], sp.mean(axis=1), acc, yaw_rate, curv, path,
             nvalid, np.minimum(nearest, 100.0), mean_nsp,
             reg_end, reg_curv, reg_end_y, disagree, cvca,
             np.abs(reg_end - straight)]
    names = ["v0", "v_mean", "accel", "yaw_rate", "curv", "path",
             "n_neigh", "nearest", "neigh_speed",
             "reg_end_dist", "reg_curv", "reg_end_y", "cv_disagree", "cv_ca_gap",
             "reg_vs_straight"]
    F = np.stack(feats, axis=1).astype(np.float32)
    if tags is not None:
        F = np.concatenate([F, tags.astype(np.float32)], axis=1)
        names = names + [f"tag_{i}" for i in range(tags.shape[1])]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, names


# --------------------------------------------------------------- 打标
@torch.no_grad()
def label_split(ds, models, split, n, K, dev, seed=0):
    rng = np.random.default_rng(seed + split)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    items = [ds[int(i)] for i in idx]
    out = {"q": {}, "cand": {}, "batch": None, "idx": idx}
    allb = {k: torch.stack([it[k] for it in items]).to(dev) for k in
            ("hist", "neigh", "neigh_mask", "future")}
    allb["tags"] = torch.stack([it["tags"] for it in items]).to(dev)
    gt = ds.denorm_future(allb["future"].cpu().numpy())
    for o in OBJS:
        qs, cands, preds = [], [], []
        K_ = 1 if o == "reg" else K
        for s0 in range(0, len(items), 256):
            b = {k: v[s0:s0 + 256] for k, v in allb.items()}
            p, _ = models[o].sample(b, K=K_, steps=STEPS[o])
            p = ds.denorm_future(p.cpu().numpy())
            preds.append(p)
            ade = np.linalg.norm(p - gt[s0:s0 + 256][:, None], axis=-1).mean(-1)
            cands.append(ade)                                # (n,K) 逐候选 ADE
            qs.append(ade.min(-1))                           # minADE@K
        out["q"][o] = np.concatenate(qs)
        out["cand"][o] = np.concatenate(cands)               # 供"诚实 oracle"检验用
        if o == "reg":
            out["reg_pred"] = np.concatenate(preds)
        print(f"    {o:5s} labelled  mean minADE {out['q'][o].mean():.3f}  "
              f"median {np.median(out['q'][o]):.3f}")
    out["gt"] = gt
    return out, allb


# --------------------------------------------------------------- 逻辑回归（不引入 sklearn）
def fit_logreg(X, y, l2=1e-2, iters=4000, lr=0.5):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xn = (Xt - mu) / sd
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        loss = loss + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return dict(w=w.detach(), b=b.detach(), mu=mu, sd=sd)


def predict_logreg(par, X):
    Xn = (torch.from_numpy(X) - par["mu"]) / par["sd"]
    return torch.sigmoid(Xn @ par["w"] + par["b"]).numpy()


def auroc(score, y):
    """ROC-AUC，用于判断门是否有区分力（0.5 = 无信息）。"""
    order = np.argsort(score)
    y = y[order]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------- 岭回归（直接预测每个模型的误差）
def fit_ridge(X, y, l2=1.0):
    """闭式解岭回归。返回标准化参数与系数。"""
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xn = (Xt - mu) / sd
    A = Xn.T @ Xn + l2 * torch.eye(Xn.shape[1])
    w = torch.linalg.solve(A, Xn.T @ (yt - yt.mean()))
    return dict(w=w, b=yt.mean(), mu=mu, sd=sd)


def predict_ridge(par, X):
    Xn = (torch.from_numpy(X) - par["mu"]) / par["sd"]
    return (Xn @ par["w"] + par["b"]).numpy()


# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/parick/dsh_ml/runs")
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--suffix", default="_lp2")
    ap.add_argument("--label-split", type=int, default=1)
    ap.add_argument("--label-n", type=int, default=12000)
    ap.add_argument("--test-split", type=int, default=2)
    ap.add_argument("--test-n", type=int, default=4096)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--taus", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0")
    ap.add_argument("--out", default="/home/parick/dsh_ml/eval/selector.json")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    models = {o: load(a.runs, o, dev, a.suffix) for o in OBJS}
    print(f"device={dev}  suffix='{a.suffix}'  K={a.K}")

    # ---------- 1. 打标
    t0 = time.time()
    print(f"\n[1] 打标 split {a.label_split}（n={a.label_n}）")
    ds1 = MiniTrajDataset(a.root, split=a.label_split, cache_logs=2)
    L, B1 = label_split(ds1, models, a.label_split, a.label_n, a.K, dev, seed=0)
    print(f"    用时 {time.time()-t0:.1f} s")

    print(f"\n[2] 打标 split {a.test_split}（n={a.test_n}）")
    ds2 = MiniTrajDataset(a.root, split=a.test_split, cache_logs=2)
    T, B2 = label_split(ds2, models, a.test_split, a.test_n, a.K, dev, seed=0)

    # ---------- 2. go / no-go
    print("\n" + "=" * 78)
    print("[3] go / no-go 关卡（协议 §6.0）")
    print("=" * 78)
    qmat = np.stack([L["q"][o] for o in OBJS], axis=1)             # (N,3)
    best_single = qmat.mean(axis=0).min()
    oracle = qmat.min(axis=1).mean()
    gap = best_single - oracle
    gap_pct = gap / best_single * 100
    print(f"  split {a.label_split} 上三条固定臂的平均 minADE: " +
          "  ".join(f"{o}={qmat.mean(axis=0)[i]:.3f}" for i, o in enumerate(OBJS)))
    print(f"  固定用最好的单模型      : {best_single:.3f} m")
    print(f"  每场都选最优（oracle）  : {oracle:.3f} m")
    print(f"  → **oracle 差距 {gap:.3f} m（{gap_pct:.1f}%）**")
    # 分歧率：贵模型真的赢了的场次比例
    for i, o in enumerate(OBJS):
        if o == "reg":
            continue
        win = qmat[:, i] < qmat[:, 0] - 1e-9
        print(f"  {o:5s} 严格优于回归的场次占比: {win.mean():.3f}")
    print(f"  fm 与 ddpm 谁更好: fm {(qmat[:,1] < qmat[:,2]).mean():.3f} / "
          f"ddpm {(qmat[:,2] < qmat[:,1]).mean():.3f}")
    if gap_pct < 2.0:
        print("\n  ⚠️ oracle 差距很小 —— 选择器的收益上限有限，必须如实报告。")
    else:
        print(f"\n  ✅ oracle 差距 {gap_pct:.1f}% —— 选择器有明确收益上限，值得做。")

    # ---------- 3. 特征
    F1, names = build_features(B1, ds1, L["reg_pred"])
    F2, _ = build_features(B2, ds2, T["reg_pred"])
    # 标准化统计只用 split 1
    print(f"\n[4] 特征 {F1.shape[1]} 维（{len(names)} 个来源），"
          f"train {F1.shape[0]} / test {F2.shape[0]}")

    # ---------- 4. τ 扫描 + 级联门
    print("\n" + "=" * 78)
    print("[5] τ 扫描 + 级联二分类门（split 1 上训练，split 2 上评估）")
    print("=" * 78)
    q1 = np.stack([L["q"][o] for o in OBJS], axis=1)
    q2 = np.stack([T["q"][o] for o in OBJS], axis=1)
    cost = np.array([NFE[o] for o in OBJS], dtype=float)
    lat = np.array([LATENCY_MS[o] for o in OBJS], dtype=float)

    hdr = (f"{'tau':>5}{'reg 够好':>9}{'fm 够好':>8}{'AUC_A':>7}{'AUC_B':>7}"
           f"{'策略 ADE':>10}{'满足率':>8}{'平均NFE':>9}{'延迟ms':>8}"
           f"{'| 全reg':>9}{'全fm':>8}{'全ddpm':>8}{'oracle':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tau in [float(x) for x in a.taus.split(",")]:
        yA = (q1[:, 0] <= tau).astype(np.float32)
        yB = (q1[:, 1] <= tau).astype(np.float32)
        if yA.sum() < 20 or yB.sum() < 20:
            continue
        gA = fit_logreg(F1, yA)
        gB = fit_logreg(F1, yB)
        pA = predict_logreg(gA, F2) > 0.5
        pB = predict_logreg(gB, F2) > 0.5
        pick = np.where(pA, 0, np.where(pB, 1, 2))       # 0=reg 1=fm 2=ddpm
        q_sel = q2[np.arange(len(pick)), pick]
        ade = q_sel.mean()
        sat = (q_sel <= tau).mean()
        nfe = cost[pick].mean()
        latm = lat[pick].mean()
        rows.append(dict(tau=tau, ade=float(ade), sat=float(sat), nfe=float(nfe),
                         lat=float(latm), mix={o: float((pick == i).mean())
                                               for i, o in enumerate(OBJS)},
                         aurocA=auroc(predict_logreg(gA, F2), (q2[:, 0] <= tau)),
                         aurocB=auroc(predict_logreg(gB, F2), (q2[:, 1] <= tau))))
        print(f"{tau:>5.2f}{yA.mean():>9.3f}{yB.mean():>8.3f}"
              f"{rows[-1]['aurocA']:>7.3f}{rows[-1]['aurocB']:>7.3f}"
              f"{ade:>10.3f}{sat:>8.3f}{nfe:>9.1f}{latm:>8.1f}"
              f"{q2[:,0].mean():>9.3f}{q2[:,1].mean():>8.3f}"
              f"{q2[:,2].mean():>8.3f}{q2.min(axis=1).mean():>8.3f}")

    # ---------- 5b. 方案 B：直接回归每个模型的误差，再按代价惩罚选
    #
    # 方案 A（上面的级联二分类门）把连续的质量信息压成一个阈值标签，浪费了信息，
    # 而且 AUC 只有 0.62–0.71，最终必然退化成"永远用最强的那个"。
    # 方案 B 改为：用岭回归预测 log q_m(i)（三个模型各一个），然后
    #     m*(i) = argmin_m [ q̂_m(i) + λ · ĉ(m) ]
    # 扫 λ 直接得到一族"精度–延迟"工作点。这同时利用了全部标签信息，
    # 也是代价敏感模型选择的标准做法。
    print("\n" + "=" * 78)
    print("[5b] 方案 B：直接回归各模型误差 + 代价惩罚（λ 扫描）")
    print("=" * 78)
    logq1 = np.log(np.maximum(q1, 1e-3))
    logq2 = np.log(np.maximum(q2, 1e-3))
    ridge = [fit_ridge(F1, logq1[:, i]) for i in range(3)]
    pred_log = np.stack([predict_ridge(ridge[i], F2) for i in range(3)], axis=1)
    # 拟合质量：预测的 log q 与实际 log q 的相关性
    for i, o in enumerate(OBJS):
        r = np.corrcoef(pred_log[:, i], logq2[:, i])[0, 1]
        print(f"  {o:5s} log q 预测与真值的相关系数 r = {r:+.3f}"
              f"   （越高说明特征越能预判该模型的误差）")

    cnorm = cost / cost.max()                 # 归一化代价 ∈ {0.011, 0.267, 1.0}
    hdr2 = (f"{'lambda':>8}{'策略 ADE':>10}{'满足率@1.5':>11}{'平均NFE':>9}{'延迟ms':>8}"
            f"{'混用 reg/fm/ddpm':>22}")
    print(hdr2)
    print("-" * len(hdr2))
    rowsB = []
    for lam in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5):
        pick = np.argmin(pred_log + lam * cnorm[None, :], axis=1)
        q_sel = q2[np.arange(len(pick)), pick]
        mix = "/".join(f"{(pick == i).mean()*100:3.0f}" for i in range(3))
        rowsB.append(dict(lam=lam, ade=float(q_sel.mean()),
                          sat=float((q_sel <= 1.5).mean()),
                          nfe=float(cost[pick].mean()),
                          lat=float(lat[pick].mean()), mix=mix))
        print(f"{lam:>8.2f}{q_sel.mean():>10.3f}{(q_sel<=1.5).mean():>11.3f}"
              f"{cost[pick].mean():>9.1f}{lat[pick].mean():>8.1f}{mix:>22}")

    # ---------- 6. Pareto 判断
    print("\n" + "=" * 78)
    print("[6] 策略是否真的占优？（精度 vs 延迟的 Pareto）")
    print("=" * 78)
    base = [("全 reg", q2[:, 0].mean(), lat[0], NFE["reg"]),
            ("全 fm", q2[:, 1].mean(), lat[1], NFE["fm"]),
            ("全 ddpm", q2[:, 2].mean(), lat[2], NFE["ddpm"])]
    for nm, ade, lm, nf in base:
        print(f"  {nm:8s} ADE {ade:6.3f}  延迟 {lm:6.1f} ms   NFE {nf:>3}")

    def pareto_check(points, tag):
        """对每个工作点，在同延迟处与固定臂的线性插值比较。"""
        print(f"\n  {tag}")
        bp = [(nm, ade, lm) for nm, ade, lm, _ in base]
        best_gain = None
        for r in points:
            ade, lm = r["ade"], r["lat"]
            cand = None
            for (n0, a0, l0), (n1, a1, l1) in [(bp[0], bp[1]), (bp[1], bp[2]),
                                              (bp[0], bp[2])]:
                if abs(l1 - l0) < 1e-9:
                    continue
                t = (lm - l0) / (l1 - l0)
                if 0 <= t <= 1:
                    v = a0 + t * (a1 - a0)
                    cand = v if cand is None else min(cand, v)
            if cand is None:
                continue
            gain = (cand - ade) / cand * 100
            if best_gain is None or gain > best_gain[0]:
                best_gain = (gain, r, cand)
        if best_gain:
            gain, r, cand = best_gain
            print(f"    最佳工作点: ADE {r['ade']:.3f} @ {r['lat']:.1f} ms，"
                  f"同延迟插值 {cand:.3f}  → 相对改善 {gain:+.1f}%")
            print(f"    {'✅ 优于固定臂的线性插值' if gain > 0.5 else '❌ 没有优于线性插值'}")
        return best_gain

    pareto_check(rows, "方案 A（级联二分类门）")
    pareto_check(rowsB, "方案 B（回归 + 代价惩罚）")

    rowsA = list(rows)                     # 方案 A 的工作点（保持独立，别和 B 混在一起）
    rowsB = rowsB
    rows_all = rowsA + [dict(r, tau=None, name="B") for r in rowsB]

    # ---------- 7. oracle 差距里有多少是"选择噪声"？用**分半诚实 oracle**
    #
    # 朴素的 oracle（逐场取 min）必然低于最好的单模型，即使三条臂完全相同 ——
    # 因为逐场取 min 会把下尾噪声也选进去。所以 22% 不能直接当收益上限。
    #
    # 正确做法：把每条臂的 K=6 条候选**分成两半**（各 3 条）。
    #   · 用 A 半选模型： m*(i) = argmin_m q_A(i,m)
    #   · 用 B 半评估所选模型： q_B(i, m*(i))
    # 若"选模型"这件事只是拟合了 A 半的噪声，收益会在 B 半上消失；
    # 若真实存在逐场差异，收益会保留。这才是"可获得收益上限"的无偏估计。
    #
    # ⚠ 之前用过的置换检验是**错的口径**：独立置换会摧毁三条臂之间 0.78–0.81 的相关性，
    #   得到的 null 差距（52.8%）比观测值（22.4%）还大，结论会完全反掉。
    print("\n" + "=" * 78)
    print("[7] 上限、实现程度、以及为什么拿不到上限")
    print("=" * 78)
    # ---- (7a) 上限：完美逐场选择（oracle）在**测试集**上的收益
    fixed_test = q2.mean(axis=0).min()
    oracle_test = q2.min(axis=1).mean()
    ceiling = fixed_test - oracle_test
    ceiling_pct = ceiling / fixed_test * 100
    print("  (7a) 理论上限（完美逐场选择，即 oracle）")
    print(f"    测试集上最好的固定模型 : {fixed_test:.3f} m   "
          f"({OBJS[int(np.argmin(q2.mean(axis=0)))]})")
    print(f"    完美逐场选择           : {oracle_test:.3f} m")
    print(f"    → **上限收益 {ceiling:.3f} m = {ceiling_pct:.1f}%**")
    print(f"    （这也是 split {a.label_split} 上的口径，两处一致："
          f"{gap:.3f} m / {gap_pct:.1f}%）")

    # ---- (7b) 我们实际拿到多少
    # ⚠ 必须在**同延迟**下比较：不同工作点的延迟差 50 倍，
    #   直接拿它的 ADE 去和"最好的固定模型"比是没有意义的。
    bp2 = [(nm, ade, lm) for nm, ade, lm, _ in base]
    def interp_at(lm):
        best = None
        for (n0, a0, l0), (n1, a1, l1) in [(bp2[0], bp2[1]), (bp2[1], bp2[2]),
                                          (bp2[0], bp2[2])]:
            if abs(l1 - l0) < 1e-9:
                continue
            t = (lm - l0) / (l1 - l0)
            if 0 <= t <= 1:
                v = a0 + t * (a1 - a0)
                best = v if best is None else min(best, v)
        return best

    print("\n  (7b) 我们的选择器拿到多少（都在**同延迟**下与固定臂线性插值比）")
    for tag, pts in (("级联二分类门", rowsA), ("代价敏感回归", rowsB)):
        if not pts:
            continue
        best, gain = None, None
        for r in pts:
            iv = interp_at(r["lat"])
            if iv is None:
                continue
            g = iv - r["ade"]
            if gain is None or g > gain:
                gain, best = g, (r, iv)
        if best is None:
            continue
        r, iv = best
        print(f"    {tag:10s}: {r['ade']:.3f} m @ {r['lat']:.1f} ms，"
              f"同延迟插值 {iv:.3f} m → 改善 {gain:.3f} m "
              f"= 上限的 {max(0.0, gain)/max(ceiling,1e-9)*100:.0f}%")

    # ---- (7c) 为什么拿不到：可预测性 + 抽样噪声
    #
    # 这里不再用"把候选分半、看 oracle 差距保留多少"的做法 —— 那个检验是错的：
    #   它把每条臂的 K 拆成 K/2，既改变了每条臂本身的质量，又让统计量更噪，
    #   得到的"保留率"没有明确含义。而且它回答的是"逐场优势是否可复现"，
    #   不是"上限是否存在" —— 上限（oracle）按定义是存在的。
    #
    # 正确的诊断拆成两项：
    #   (i)  可预测性：用特征预测 log q 的相关性 r，以及逐场**优势**的相关性；
    #   (ii) 抽样可靠性：把每条臂的 K 条候选随机分成两半，
    #        两半各自算出的逐场优势之间的一致性 —— 这部分是任何选择器都学不到的噪声。
    print("\n  (7c) 为什么拿不到上限 —— 两个独立的瓶颈")
    print("    (i) 可预测性（特征能不能预判'哪个模型更好'）")
    for i, o in enumerate(OBJS):
        r = np.corrcoef(pred_log[:, i], logq2[:, i])[0, 1]
        print(f"        {o:5s}: 预测 log q 与真值 r = {r:+.3f}  (R² = {r*r:.2f})")
    # 逐场优势的可预测性（reg vs fm、fm vs ddpm 各一条）
    for i, j in ((0, 1), (1, 2)):
        adv_true = logq2[:, i] - logq2[:, j]
        adv_pred = pred_log[:, i] - pred_log[:, j]
        r = np.corrcoef(adv_pred, adv_true)[0, 1]
        print(f"        优势 ({OBJS[i]}−{OBJS[j]}) 的可预测性 r = {r:+.3f}")

    print("    (ii) 抽样可靠性（逐场优势里有多少是采样噪声，谁都学不到）")
    rng2 = np.random.default_rng(1)
    for i, j in ((0, 1), (1, 2)):
        rs = []
        for _ in range(20):
            ha, hb = [], []
            for k in (i, j):
                cd = T["cand"][OBJS[k]]
                if cd.shape[1] < 2:
                    ha.append(cd[:, 0]); hb.append(cd[:, 0]); continue
                perm = rng2.permutation(cd.shape[1])
                h = cd.shape[1] // 2
                ha.append(cd[:, perm[:h]].min(-1))
                hb.append(cd[:, perm[h:]].min(-1))
            aa = ha[0] - ha[1]
            bb = hb[0] - hb[1]
            rs.append(np.corrcoef(aa, bb)[0, 1])
        print(f"        优势 ({OBJS[i]}−{OBJS[j]}) 两半候选之间的一致性 r = "
              f"{np.mean(rs):+.3f} ± {np.std(rs):.3f}")
    print("\n    判读：")
    print("      · (i) 低 → 特征不够，换更好的特征（地图/车道）直接受益，这是主要瓶颈。")
    print("      · (ii) 高 → 逐场差异本身是真实的，不是采样噪声，值得继续投入。")
    print("      · 两者都低 → 说明这个任务上'逐场换模型'的空间确实有限。")

    # ---------- 保存
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(label_split=a.label_split, label_n=int(L["idx"].size),
                       test_split=a.test_split, test_n=int(T["idx"].size),
                       K=a.K, nfe=NFE, latency_ms=LATENCY_MS,
                       oracle_gap=dict(best_single=float(best_single),
                                       oracle=float(oracle), gap=float(gap),
                                       gap_pct=float(gap_pct),
                                       means={o: float(qmat.mean(axis=0)[i])
                                              for i, o in enumerate(OBJS)}),
                       fixed={o: float(np.stack([T["q"][x] for x in OBJS], 1)
                                       .mean(axis=0)[i]) for i, o in enumerate(OBJS)},
                       oracle_test=float(q2.min(axis=1).mean()),
                       sweep=rows_all), fh, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {a.out}")


if __name__ == "__main__":
    main()
