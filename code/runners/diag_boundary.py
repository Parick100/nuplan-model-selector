"""验证审查提出的 RISK-2：位置 DCT 低通在**端点**有系统性偏差。

审查意见：DCT-II 对 n 个点做偶延拓，而"从原点出发的斜坡"在边界处**导数不连续**
（偶延拓后是个折线），截断到前 M 个模式会在两端产生 Gibbs 式误差。
审查给的量级：常数速度轨迹在两端各有 0.274·(v·dt) 的偏差 —— v=15 m/s 时 0.41 m。
这个偏差在"99.9996% 能量保留"的平均统计里**看不见**，但它是**系统性**的。

对照方案（审查建议）：约束**速度剖面**而不是位置剖面 ——
把一阶差分投影到前 M 个模式，再从锚点积分回去：

    p ← L · P_v · L⁻¹ p        （L = 下三角累积和）

这样位置自动从锚点出发、导数连续，端点的折角消失。

本脚本只做代数比较，不训练：
  (1) 常数速度轨迹（解析可算）下两种算子的端点偏差随速度的变化
  (2) 真实真值上两种算子的残差**随时间的分布**（是否集中在首尾）
  (3) 两种算子各自的 minADE / minFDE 地板（loss 最优解可达的精度下界）
  (4) 两种算子是否都仍幂等、秩相同

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/diag_boundary.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
sys.path.insert(0, os.path.join(HERE, "..", "models"))

from dataset import MiniTrajDataset                     # noqa: E402
from model import dct_lowpass                           # noqa: E402

ROOT = "/home/parick/dsh_ml/cache"
N, DT = 80, 0.1


def proj_pos(m):
    """位置低通：p -> P p（当前实现）。返回 (N,N)。"""
    return dct_lowpass(N, m).numpy().astype(np.float64)


def proj_vel(m):
    """速度低通：p -> C · P_v · A · p（审查建议的方案）。返回 (N,N)。

    关键：必须把**锚点**一起纳入差分。序列是 81 点 `[0, p₀, …, p₇₉]`，
    一阶差分有 80 个分量（正好与未来点数相同）：

        A (80×80)  p ↦ Δ，其中 Δ₀ = p₀ − 0，Δᵢ = pᵢ − pᵢ₋₁   （下三角，对角 1）
        P_v        把 Δ 投影到前 M 个 DCT 模式（长度 80）
        C (80×80)  Δ ↦ 累积和，从锚点的 0 出发                （下三角全 1）

    因为 C·A = I（一阶差分再累积和是恒等），所以

        P_vel = C P_v A   ⇒   P_vel² = C P_v (A C) P_v A = C P_v² A = P_vel

    仍是幂等投影，秩 = M。**常数速度轨迹的 Δ 是常向量**（全部落在第 0 个模式上），
    所以 P_vel 对匀速轨迹是**精确恒等** —— 这正是位置低通做不到的：
    位置低通对匀速斜坡在两端有 0.274·(v·dt) 的 Gibbs 偏差。

    第一次实现漏掉了锚点（只在 80 点未来上做 L·P·L⁻¹），于是 L·D = I − p₀·1
    不是恒等，算子既不幂等、又在末点产生整整 1.0·(v·dt) 的偏移。教训：
    "积分回去"必须从**已知的锚点**开始，而不是从数组第一个元素开始。
    """
    Pv = dct_lowpass(N, m).numpy().astype(np.float64)     # 80 个差分，80 个模式
    A = np.eye(N) - np.diag(np.ones(N - 1), -1)           # Δᵢ = pᵢ − pᵢ₋₁
    C = np.tril(np.ones((N, N)))                          # 累积和
    return C @ Pv @ A


def report_operator(name, P, gt, cv_speeds):
    print(f"\n--- {name} ---")
    Id = np.eye(N)
    print(f"  幂等 |P²−P| = {np.abs(P @ P - P).max():.2e}   "
          f"对称 {np.abs(P - P.T).max():.2e}   秩 {np.linalg.matrix_rank(P)}   "
          f"trace {np.trace(P):.4f}")

    # (1) 常数速度轨迹的端点偏差
    print("  常数速度轨迹（p_k = v·dt·k）的逐点偏差 |P p − p|：")
    print(f"    {'v (m/s)':>8} {'首点偏差':>10} {'末点偏差':>10} {'最大偏差':>10} {'偏差/(v·dt)':>13}")
    for v in cv_speeds:
        p = (v * DT * np.arange(1, N + 1))[:, None]
        r = (P @ p) - p
        print(f"    {v:>8.1f} {abs(r[0,0]):>10.4f} {abs(r[-1,0]):>10.4f} "
              f"{np.abs(r).max():>10.4f} {np.abs(r).max()/(v*DT):>13.4f}")

    # (2) 真实真值上残差的时间分布
    r = np.einsum("ij,bjc->bic", P, gt) - gt
    rm = np.linalg.norm(r, axis=-1)                        # (B, N)
    prof = rm.mean(axis=0)
    print(f"  真实真值（{len(gt)} 条）：平均逐点残差 "
          f"首点 {prof[0]:.4f}  末点 {prof[-1]:.4f}  中段均值 {prof[20:60].mean():.4f} m")
    print(f"    整体 mean {rm.mean():.4f}  max {rm.max():.4f} m")
    # 偏差 vs 噪声：残差的均值向量（有符号）比模长更能说明"系统性"
    bias = np.linalg.norm(r.mean(axis=0), axis=-1)
    print(f"    有符号偏差 |mean_batch(r)| 首点 {bias[0]:.4f}  末点 {bias[-1]:.4f}  "
          f"中段均值 {bias[20:60].mean():.4f} m")
    print(f"    偏置占比（有符号偏差模长 / 残差模长）= "
          f"{bias.mean()/max(rm.mean(),1e-9):.3f}")

    # (3) 地板：若模型只能输出 P 作用后的结果，最好的 minADE/minFDE 是多少
    #     loss 最优解 = P gt（在约束集内离 gt 最近），minADE 就是它的平均误差
    Pg = np.einsum("ij,bjc->bic", P, gt)
    ade = np.linalg.norm(Pg - gt, axis=-1).mean()
    fde = np.linalg.norm(Pg[:, -1] - gt[:, -1], axis=-1).mean()
    print(f"  loss 最优解 P·gt 对真值的 minADE 地板 {ade:.4f} m   minFDE 地板 {fde:.4f} m")
    # 自由地找约束集内使 minADE 最小的点（逐条沿时间平均误差最小化）
    return {"ade_floor": ade, "fde_floor": fde, "max_bias": bias.max(),
            "mean_resid": rm.mean()}


def main():
    ds = MiniTrajDataset(ROOT, split=2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=2000, replace=False)
    gt = np.stack([ds.denorm_future(ds[int(i)]["future"].numpy()) for i in idx])
    print(f"真值 {gt.shape}（split 2，米制）")
    print(f"真值末点距离 mean {np.linalg.norm(gt[:,-1],axis=-1).mean():.2f} m "
          f"→ 等效平均速度 {np.linalg.norm(gt[:,-1],axis=-1).mean()/(N*DT):.2f} m/s")

    M = 24
    a = report_operator(f"方案 A：位置低通 P_pos（当前实现，M={M}）",
                        proj_pos(M), gt, (5, 10, 15, 20, 30))
    b = report_operator(f"方案 B：速度低通 L·P_v·L⁻¹（审查建议，M={M}）",
                        proj_vel(M), gt, (5, 10, 15, 20, 30))

    print("\n=== 结论 ===")
    for k, lbl in (("mean_resid", "平均残差 (m)"), ("max_bias", "最大有符号偏差 (m)"),
                   ("ade_floor", "minADE 地板 (m)"), ("fde_floor", "minFDE 地板 (m)")):
        print(f"  {lbl:22s} A(位置) {a[k]:8.4f}   B(速度) {b[k]:8.4f}   "
              f"改善 {(a[k]-b[k])/max(a[k],1e-12)*100:6.1f}%")


if __name__ == "__main__":
    main()
