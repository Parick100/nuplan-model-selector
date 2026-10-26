"""W2 — 指标与计时。口径以 EXPERIMENTS_实验协议.md §4 为准。

张量约定（全部 numpy）：
    preds : (N, K, T, 2)   K 条候选，T 个时间点
    gt    : (N, T, 2)

指标
    minADE@K / minFDE@K / MissRate@K
    diversity            候选集内部的平均两两 FDE（多模态覆盖）
    coverage_nll         真值在 K 条样本 KDE 下的对数似然（低维投影，避免高维病态）
    kinematic_violation  加速度 / jerk 超限率（替代 off-road，因地图暂缺）
"""
import numpy as np

DT = 0.1                       # 10 Hz
MISS_THR = 2.0                 # m，minFDE 超过它算 miss
A_MAX = 4.0                    # m/s^2 舒适上界（nuPlan 附近）
J_MAX = 8.0                    # m/s^3
SIGMA_FLOOR = 1.0              # m，KDE 带宽下限（见 coverage_nll 注释）
NLL_CAND_FLOOR = 2.0           # m，单候选（回归基线）时的固定尺度


# ----------------------------------------------------------------- 精度
def _pairwise_dist(preds, gt):
    """(N,K,T) 欧氏距离。"""
    return np.linalg.norm(preds - gt[:, None, :, :], axis=-1)


def min_ade(preds, gt, k=None):
    d = _pairwise_dist(preds, gt)                 # (N,K,T)
    if k is not None:
        d = d[:, :k]
    return d.mean(axis=-1).min(axis=-1)           # (N,)


def min_fde(preds, gt, k=None):
    d = np.linalg.norm(preds[:, :, -1, :] - gt[:, None, -1, :], axis=-1)   # (N,K)
    if k is not None:
        d = d[:, :k]
    return d.min(axis=-1)


def miss_rate(preds, gt, k=None, thr=MISS_THR):
    return (min_fde(preds, gt, k) > thr).mean()


def diversity(preds, k=None):
    """每条场景内候选两两 FDE 的均值，再对场景取平均。单候选时返回 0。"""
    p = preds if k is None else preds[:, :k]
    N, K = p.shape[0], p.shape[1]
    if K < 2:
        return np.zeros(N)
    end = p[:, :, -1, :]                          # (N,K,2)
    tot = np.zeros(N)
    cnt = 0
    for i in range(K):
        for j in range(i + 1, K):
            tot += np.linalg.norm(end[:, i] - end[:, j], axis=-1)
            cnt += 1
    return tot / cnt


# ----------------------------------------------------------------- 覆盖
def _project(x):
    """(N,K,T,2) 或 (N,T,2) -> 低维表示。用末位移 + 中点位移（4 维）。"""
    mid = x[..., x.shape[-2] // 2, :]
    end = x[..., -1, :]
    return np.concatenate([end, mid], axis=-1)


def coverage_nll(preds, gt, k=None, eps=1e-6, sigma_floor=SIGMA_FLOOR):
    """真值在候选集 KDE 下的对数似然（4 维投影，对角高斯核）。

    K>=2 用样本协方差 + Scott 带宽（h = K^(-1/(D+4))）；K==1 退化为固定尺度的
    单峰高斯，这样回归基线（单输出）与生成模型可以用同一口径对比。

    ⚠ 带宽下限 sigma_floor 不可省：当 K 条候选几乎重合时（例如模型坍缩成单模态），
    样本协方差→0，似然会数值爆炸到 -1e6 量级，指标失去意义。下限取 1 m，
    与 miss-rate 阈值 2 m 属同一量级的设计常数，必须在报告里写明。
    """
    p = _project(preds if k is None else preds[:, :k])      # (N,K,D)
    g = _project(gt)                                        # (N,D)
    N, K, D = p.shape
    if K < 2:
        sigma = np.full(D, NLL_CAND_FLOOR)
        diff = (g - p[:, 0]) / sigma
        return -0.5 * (diff ** 2).sum(-1) - np.log(sigma).sum() - 0.5 * D * np.log(2 * np.pi)
    out = np.empty(N)
    h = K ** (-1.0 / (D + 4.0))                             # Scott 规则
    for n in range(N):
        s = p[n]
        cov = np.cov(s.T) + eps * np.eye(D)
        sig = np.sqrt(np.clip(np.diag(cov), eps, None)) * h
        sig = np.maximum(sig, sigma_floor)                  # ← 关键：防止坍缩
        z = (g[n] - s) / sig                                # (K,D)
        logk = -0.5 * (z ** 2).sum(-1) - np.log(sig).sum() - 0.5 * D * np.log(2 * np.pi)
        m = logk.max()
        out[n] = m + np.log(np.exp(logk - m).mean())
    return out


# ----------------------------------------------------------------- 运动学
#
# ⚠ 绝对阈值（A_MAX=4 m/s²、J_MAX=8 m/s³）在**真值自己**身上就已经 100% 超限，
#   见 code/metrics/diag_gt_noise.py 与 code/data/diag_position_jitter.py。实测：
#     · lidar_pc 是干净的 20 Hz，10 Hz 抽样后 dt = 0.1000 s ±7%，无重复行；
#     · 但跟踪位置列本身在帧间抖动，静止车 ±0.02–0.05 m，快车更大；
#     · 直接二阶差分：真值 |a| p50 = 4.7、p95 = 20.3、max = 125 m/s²；
#     · corr(Δp/dt, 数据库自带 v) = +0.99，说明抖动在**跟踪器输出里**，不是我们的 bug；
#     · 离散三阶差分再把噪声放大 1/dt³ = 1000 倍 → 真值 jerk 超限率必然 1.000。
#   一个在真值上返回 1.000 的指标零信息量。
#
#   因此改为**相对口径**：阈值取真值自身分布的 95 分位，同一个算子作用在预测与
#   真值上。真值的超限率恒为 0.05，这是"零分基准"；模型 > 0.05 表示比真实驾驶
#   更抖。同时给出模型/真值的 p95 比值，便于跨数据集比较。
A_MAX_LEGACY = 4.0             # 保留仅为兼容旧报告，不要再用
J_MAX_LEGACY = 8.0             # 同上
KIN_PCT = 95.0                 # 相对口径的分位


def _kin_pointwise(p, dt=DT, kmax=None):
    """逐点 |a| / |j|（对候选取最差）。返回 (N, T-2) 与 (N, T-3)。

    用**逐点分布**而不是"每条轨迹的峰值"：峰值是 78 个点的极值，重尾极强
    （真值 p95 的峰值达 88.7 m/s²），拿它当阈值会让指标退化成恒不触发。
    逐点分布才是有 5% 尾部、能真正区分好坏的量。
    """
    if p.ndim == 3:                                  # (N,T,2) 视为单候选
        p = p[:, None]
    q = p if kmax is None else p[:, :kmax]
    v = np.diff(q, axis=-2) / dt
    a = np.diff(v, axis=-2) / dt
    j = np.diff(a, axis=-2) / dt
    am = np.linalg.norm(a, axis=-1).max(axis=1)      # (N, T-2) 对 K 取最差
    jm = np.linalg.norm(j, axis=-1).max(axis=1)      # (N, T-3)
    return am, jm


def kinematic_violation(preds, gt=None, dt=DT, pct=KIN_PCT):
    """逐点运动学统计。返回 dict。

    给了 `gt` 时阈值取**真值逐点 |a| / |j| 的 pct 分位**（相对口径；真值基准 = 0.05）。
    未给时退回旧的绝对阈值（此时几乎必然 1.000，仅作对照）。

    键：
      accel_violation / jerk_violation  逐点超限率（真值基准 0.05）
      max_accel / max_jerk              每条轨迹峰值再对场景平均（上报用，非阈值）
      accel_p95_ratio                   模型逐点 |a| p95 / 真值逐点 |a| p95（真值 = 1.0）
    """
    am, jm = _kin_pointwise(preds, dt)
    out = {
        "accel_violation": None, "jerk_violation": None,
        "max_accel": float(am.max(axis=-1).mean()),
        "max_jerk": float(jm.max(axis=-1).mean()),
        "accel_p50": float(np.median(am)), "accel_p95": float(np.percentile(am, 95)),
        "jerk_p50": float(np.median(jm)), "jerk_p95": float(np.percentile(jm, 95)),
    }
    if gt is None:
        out.update(accel_thr=A_MAX_LEGACY, jerk_thr=J_MAX_LEGACY,
                   note="绝对阈值（真值上也会 100% 超限，仅供参考）",
                   accel_p95_ratio=float("nan"))
        out["accel_violation"] = float((am > A_MAX_LEGACY).mean())
        out["jerk_violation"] = float((jm > J_MAX_LEGACY).mean())
        return out
    g_a, g_j = _kin_pointwise(gt, dt)
    a_thr = float(np.percentile(g_a, pct))
    j_thr = float(np.percentile(g_j, pct))
    if a_thr < 1e-6:
        a_thr, note = A_MAX_LEGACY, "真值 p95=0（光滑合成数据），退回绝对阈值"
    else:
        note = f"真值逐点 p{pct:.0f} 标定"
    if j_thr < 1e-6:
        j_thr = J_MAX_LEGACY
    out.update(accel_thr=a_thr, jerk_thr=j_thr, note=note,
               accel_violation=float((am > a_thr).mean()),
               jerk_violation=float((jm > j_thr).mean()),
               accel_p95_ratio=float(np.percentile(am, 95) /
                                     max(np.percentile(g_a, 95), 1e-3)))
    return out


# ----------------------------------------------------------------- 汇总
def summarize(preds, gt, ks=(1, 4, 6, 16), thr=MISS_THR):
    """返回 {K: {...}} 的指标表。K 超过实际候选数时报 nan。"""
    out = {}
    Kmax = preds.shape[1]
    for k in ks:
        if k > Kmax:
            out[k] = {m: float("nan") for m in
                      ("minADE", "minFDE", "missRate", "diversity", "coverage_nll")}
            continue
        out[k] = {
            "minADE": float(min_ade(preds, gt, k).mean()),
            "minFDE": float(min_fde(preds, gt, k).mean()),
            "missRate": float(miss_rate(preds, gt, k, thr)),
            "diversity": float(diversity(preds, k).mean()),
            "coverage_nll": float(coverage_nll(preds, gt, k).mean()),
        }
    ks = kinematic_violation(preds, gt)
    out["kinematic"] = ks
    return out


def format_table(res, title=""):
    lines = [f"--- {title} ---"]
    lines.append(f"{'K':>4} {'minADE':>8} {'minFDE':>8} {'missRate':>9} "
                 f"{'diversity':>10} {'covNLL':>9}")
    for k, v in res.items():
        if k == "kinematic":
            continue
        lines.append(f"{k:>4} {v['minADE']:>8.3f} {v['minFDE']:>8.3f} "
                     f"{v['missRate']:>9.3f} {v['diversity']:>10.3f} "
                     f"{v['coverage_nll']:>9.3f}")
    kv = res["kinematic"]
    lines.append(f"kinematic ({kv['note']}, 逐点阈值 |a|>{kv['accel_thr']:.2f} m/s² "
                 f"|j|>{kv['jerk_thr']:.1f} m/s³，真值基准 0.05):")
    lines.append(f"  accel_viol {kv['accel_violation']:.3f}  "
                 f"jerk_viol {kv['jerk_violation']:.3f}  "
                 f"|a| p50/p95 {kv['accel_p50']:.2f}/{kv['accel_p95']:.2f}  "
                 f"峰值均值 {kv['max_accel']:.2f}  "
                 f"|a|p95/真值 {kv['accel_p95_ratio']:.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------- 自检
def _selftest():
    N, K, T = 3, 4, 80
    rng = np.random.default_rng(0)
    gt = np.zeros((N, T, 2))
    gt[:, :, 0] = np.linspace(0, 40, T)[None, :]

    # 场景0：候选0 完全命中 → minADE@K 应为 0
    p = np.zeros((N, K, T, 2))
    p[0, 0] = gt[0]
    p[0, 1:] = gt[0] + 10.0
    # 场景1：只沿 y 偏 3 m（若同时偏 x、y 距离会是 sqrt(18)）
    p[1, :] = gt[1]
    p[1, :, :, 1] += 3.0
    # 场景2：候选终点分别偏 1,2,3,4 m
    for k in range(K):
        p[2, k] = gt[2]
        p[2, k, -1, 1] += (k + 1)

    fde = min_fde(p, gt)
    assert abs(fde[0]) < 1e-9, fde
    assert abs(fde[1] - 3.0) < 1e-6, fde
    assert abs(fde[2] - 1.0) < 1e-6, fde

    ade = min_ade(p, gt)
    assert ade[0] < 1e-9
    assert ade[1] > 2.9 and ade[1] < 3.1

    # 只给前 2 条候选时，场景2 的 minFDE 应为 1
    assert abs(min_fde(p, gt, 2)[2] - 1.0) < 1e-6

    # diversity：场景2 终点 1,2,3,4 → 平均两两距离 = (1+2+3 +1+2 +1)/6 = 10/6
    dv = diversity(p, None)[2]
    assert abs(dv - 10.0 / 6.0) < 1e-6, dv

    # miss rate：阈值 2.0，场景1 的 minFDE=3 → miss；场景0 不 miss
    assert abs(miss_rate(p, gt, None, 2.0) - 1.0 / 3.0) < 1e-9

    # coverage_nll：完全命中的候选集，NLL 应显著高于全偏 10 m 的
    n = coverage_nll(p, gt)
    assert n[0] > n[1], (n[0], n[1])
    # 关键回归：候选重合（坍缩）时不得出现数值爆炸
    assert np.all(n > -60.0), f"NLL 爆炸: {n}"
    p_deg = np.repeat(gt[:, None, :, :], 4, axis=1)         # 4 条与真值完全相同的候选
    nd = coverage_nll(p_deg, gt)
    assert np.all(np.abs(nd) < 60.0), f"退化候选 NLL 异常: {nd}"

    ks = kinematic_violation(p, gt)
    assert isinstance(ks["accel_p95"], float) and 0.0 <= ks["accel_violation"] <= 1.0
    assert ks["max_accel"] > 0.0, "合成数据里候选偏了 10 m，峰值加速度必须为正"
    # 相对口径的关键性质：**真值自己**必须接近 p95 基准 0.05，而不是 1.000。
    # 这是"指标有没有信息量"的回归测试 —— 旧实现会在真值上返回 1.000。
    # 必须用**有帧间抖动**的真值：完全光滑的合成直线其 |a| 分位数为 0，
    # 会走"退回绝对阈值"的分支，测不到这条性质（这正是 nuPlan 反而好测的地方）。
    rng2 = np.random.default_rng(3)
    gtj = gt + rng2.normal(0.0, 0.05, gt.shape)          # 5 cm 位置抖动
    gv = kinematic_violation(gtj[:, None], gtj)
    r = gv["accel_violation"]
    assert gv["accel_thr"] > A_MAX_LEGACY, "5 cm 抖动下应走真值标定分支，而非绝对阈值回退"
    assert abs(r - (100 - KIN_PCT) / 100.0) < 0.02, f"真值超限率应≈0.05，实为 {r}"
    assert abs(gv["accel_p95_ratio"] - 1.0) < 1e-6, "真值对自己的 |a|p95 比应为 1.0"
    print(f"回归测试：真值自身加速度超限率 {r:.3f}（应 ≈0.05，旧实现是 1.000）")

    print("self-test PASSED")
    print(format_table(summarize(p, gt, ks=(1, 2, 4)), "synthetic"))


if __name__ == "__main__":
    _selftest()
