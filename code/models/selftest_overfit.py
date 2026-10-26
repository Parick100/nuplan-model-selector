"""W3 验收测试：单 batch 过拟合（memorize one batch）。

这是验证生成模型实现是否正确的**唯一可靠手段**：一个实现正确的扩散/流匹配
模型必须能把一个小 batch 的 loss 压到接近 0，并且**采样能复现这批真值**。

它能抓到的典型错误：
- 时间方向搞反（从 t=0 积到 t=1 而不是 t=1 到 t=0）
- ᾱ(t) 公式写错、√ 与 ᾱ 搞混
- 流匹配目标写成 x₀−ε 而不是 ε−x₀（现已统一成 x0 参数化，见 model.py 注释 3）
- 采样步长符号错
- 条件向量没接进网络
- 低通投影写错（A 不幂等 / b 没配对 → 采样结果系统性偏移）

**走真实数据通路**：直接复用 `MiniTrajDataset` 与 `norm.json`，不另写一套加载，
否则测过的代码和训练的代码不是同一份。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/models/selftest_overfit.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics"))

from dataset import MiniTrajDataset                              # noqa: E402
from model import TrajectoryModel, alpha_bar                     # noqa: E402
from traj_metrics import min_ade, min_fde                        # noqa: E402

ROOT = "/home/parick/dsh_ml/cache"
BKEYS = ("hist", "neigh", "neigh_mask", "future")


def load_batch(n=8, split=0, seed=0):
    """从 split 里样本最多的那个 log 随机抽 n 条。

    ⚠ 审查抓到的 bug：`ds.index` 的元素是 `(log下标, log内行号)`，两次都当成**全局**
    下标用会取错样本 —— `rows = [j for (a, j) in ds.index if a == li]` 拿到的是 log 内
    行号，再 `ds[int(i)]` 就全落到数据集开头的 log 0 里去了，而打印的标签还写着
    另一个 log 的名字。正确做法是 `enumerate(ds.index)` 取全局下标。
    """
    ds = MiniTrajDataset(ROOT, split)
    e = max(ds.logs, key=lambda x: x["kept"])
    li = ds.logs.index(e)
    rows = [k for k, (a, _j) in enumerate(ds.index) if a == li]      # ← 全局下标
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows, size=n, replace=False) if len(rows) >= n else rows
    items = [ds[int(i)] for i in pick]
    b = {k: torch.stack([it[k] for it in items]) for k in BKEYS}
    # 自检：抽到的样本必须真的都来自同一个 log（否则说明下标又搞错了）
    logs = {ds.log_of(int(i)) for i in pick}
    assert len(logs) == 1, f"抽样跨越了 {len(logs)} 个 log: {logs}"
    assert e["logfile"] in logs, f"抽到的 log {logs} 与目标 {e['logfile']} 不符"
    return b, e["logfile"], ds


@torch.no_grad()
def field_err_by_t(m, b, nbins=6):
    """按 t 分箱的场误差：验证"训练目标本身在小 t 处更病态"这一判断。

    两种范式现在都预测 x0，所以误差应当**在全 t 上大致平稳**（这正是改 x0
    参数化的目的）。若仍然集中在小 t，说明还有参数化层面的问题。
    """
    gt = m._assemble(b["future"], b["future"].shape[0], b["future"].device)
    B = gt.shape[0]
    tokens, mask, _ = m.encode(b)
    rows = []
    for i in range(nbins):
        t = torch.full((B,), (i + 0.5) / nbins, device=gt.device)
        eps = torch.randn_like(gt)
        x = gt.reshape(B, m.out_len, 2)
        e = eps.reshape(B, m.out_len, 2)
        x = x.clone()
        x[:, 0, :] = 0.0
        e[:, 0, :] = 0.0
        if m.objective == "ddpm":
            ab = alpha_bar(t).view(B, 1, 1)
            x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * e
        else:
            tv = t.view(B, 1, 1)
            x_t = (1 - tv) * x + tv * e
        pred = m.predict_field(x_t.reshape(B, -1), t, tokens, mask)
        rows.append((float(t[0]), float(((pred - x.reshape(B, -1)) ** 2).mean())))
    return rows


def rough(x):
    """二阶差分幅度（米制），与协议 §4 的 smoothness 同口径。x: (...,T,2) numpy。"""
    a = x[..., 2:, :] - 2 * x[..., 1:-1, :] + x[..., :-2, :]
    return np.linalg.norm(a, axis=-1)


def run(obj, batch, ds, dev, steps=2500, lr=3e-3, sample_steps=15, drop=0.0,
        basis_m=24, trace_every=0, verbose=True):
    """过拟合测试。drop=0 很关键：随机深度/丢弃会让记忆任务永远收敛不到 0。"""
    torch.manual_seed(0)
    m = TrajectoryModel(objective=obj, drop=drop, basis_m=basis_m,
                        norm_mu=ds.mu, norm_sd=ds.sd).to(dev)
    b = {k: v.to(dev) for k, v in batch.items()}
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0 = time.time()
    first = last = None
    trace = []
    for i in range(steps):
        loss = m(b)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
        if trace_every and (i + 1) % trace_every == 0:
            trace.append((i + 1, last))
    dt = time.time() - t0

    # 采样复现检查（**米制**空间比较，否则量纲没有意义）
    m.eval()
    K = 8 if obj != "reg" else 1
    preds, nfe = m.sample(b, K=K, steps=sample_steps)
    p_m = ds.denorm_future(preds.cpu().numpy())                  # (B,K,80,2) 米
    g_m = ds.denorm_future(b["future"].cpu().numpy())            # (B,80,2)  米
    ade = min_ade(p_m, g_m)
    fde = min_fde(p_m, g_m)
    rough_gt = float(rough(g_m).mean())
    rough_pd = float(rough(p_m).mean())

    # ⚠ 关键：以上数字**必须与约束自身的地板比较**。
    # 审查指出：约束把输出限制在一个子空间里，真值本身不在其中，
    # 所以"模型记住了这一条"的 minADE 恰好等于 `mean|proj(gt) − gt|`。
    # 因此单看 "minADE 0.023 m" 不能证明学会了任何东西 —— 它只证明
    # **采样器复现了模型自己的 x̂0**（那仍然是非常有价值的管道验证）。
    # 把它和地板并列报出来，读者一眼就知道该看哪个数。
    with torch.no_grad():
        pg_m = ds.denorm_future(m._proj_fut(b["future"]).cpu().numpy())
    floor_ade = float(np.linalg.norm(pg_m - g_m, axis=-1).mean())
    floor_fde = float(np.linalg.norm(pg_m[:, -1] - g_m[:, -1], axis=-1).mean())

    by_t = field_err_by_t(m, b) if obj != "reg" else []

    if verbose:
        tr = "  ".join(f"{s}:{v:.2e}" for s, v in trace)
        print(f"  {obj:5s} loss {first:8.4f} -> {last:.3e}   "
              f"降 {first/max(last,1e-12):9.1f}x")
        print(f"        采样 vs 真值 : minADE {ade.mean():7.4f}  minFDE {fde.mean():7.4f} m")
        print(f"        约束的地板   : minADE {floor_ade:7.4f}  minFDE {floor_fde:7.4f} m  "
              f"← 比值 ADE {ade.mean()/max(floor_ade,1e-12):.2f}× "
              f"FDE {fde.mean()/max(floor_fde,1e-12):.2f}×（≈1 表示采样器复现了 x̂0）")
        print(f"        粗糙度 |d²p| 预测 {rough_pd:.4f} / 真值 {rough_gt:.4f} m "
              f"= {rough_pd/max(rough_gt,1e-9):7.2f}x   nfe {nfe}   {dt:5.1f}s")
        if tr:
            print(f"        trace {tr}")
        if by_t:
            print("        按 t 分箱的 x0 场误差: " +
                  "  ".join(f"t={t:.2f}:{v:.2e}" for t, v in by_t))
    return {"obj": obj, "first": first, "last": last, "trace": trace, "by_t": by_t,
            "ade": float(ade.mean()), "fde": float(fde.mean()), "nfe": nfe,
            "floor_ade": floor_ade, "floor_fde": floor_fde,
            "ade_over_floor": float(ade.mean()) / max(floor_ade, 1e-12),
            "fde_over_floor": float(fde.mean()) / max(floor_fde, 1e-12),
            "rough_ratio": rough_pd / max(rough_gt, 1e-9),
            "fde_scale": float(np.linalg.norm(g_m[:, -1], axis=-1).mean())}


def judge(res, strict=True):
    """判据（本次已收紧，见审查意见）：

    1. loss 至少降 50×；
    2. **采样结果落在约束地板上**：`minADE/minFDE ≤ max(1.5×地板, 0.05 m)`。
       这才是真正的"实现正确"判据：它同时要求 (a) 采样器精确复现模型自己的 x̂0，
       (b) loss 真的被优化到地板附近。旧判据 `minFDE < 10% × 真值尺度`（本例 2.73 m）
       比地板（0.156 m）宽 17 倍，基本测不出任何东西。

       ⚠ 两条关于容差的事必须写清：
       · **必须有绝对项**：速度空间的约束把 **minFDE 地板压到了 0.0000 m**
         （位置空间是 0.1706 m），纯比值判据会除以 0 而报出 1e4 倍的假失败。
       · **0.05 m 的依据**：本数据集自身的帧间跟踪抖动就是 ~5 cm
         （见 code/data/diag_position_jitter.py），所以没有"拟合噪声以外的"余量可压。
         本例轨迹末点 38.7 m，0.05 m 相当于它的 0.13%。
    3. 粗糙度 ≤ 3× 真值（约束本身已保证，仅作回归保护）。
    """
    TOL = 0.05                     # m，见上面 docstring 的依据
    print(f"\n=== 正确性判据：n=1 记忆（相对**约束地板**判定，绝对容差 {TOL} m）===")
    ok = True
    for r in res:
        c1 = r["first"] / max(r["last"], 1e-12) >= 50.0
        a_lim = max(1.5 * r["floor_ade"], TOL)
        f_lim = max(1.5 * r["floor_fde"], TOL)
        c2 = bool(r["ade"] <= a_lim and r["fde"] <= f_lim)
        c3 = r["rough_ratio"] <= 3.0
        passed = c1 and c2 and c3
        mark = "OK " if passed else ("!! " if strict else "·  ")
        print(f"  {mark}{r['obj']:5s} loss 降 {r['first']/max(r['last'],1e-12):8.1f}x  "
              f"minADE {r['ade']:.4f} ≤ {a_lim:.4f}  "
              f"minFDE {r['fde']:.4f} ≤ {f_lim:.4f}  "
              f"粗糙度 {r['rough_ratio']:5.2f}×   "
              f"（地板 {r['floor_ade']:.4f}/{r['floor_fde']:.4f} m，尺度 {r['fde_scale']:.1f} m）")
        if strict:
            ok &= passed
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--basis-m", type=int, default=24)
    ap.add_argument("--trace-every", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--objs", default="reg,ddpm,fm")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  steps={a.steps}  lr={a.lr}  drop=0  basis_m={a.basis_m}")

    batch, logname, ds = load_batch(a.n)
    print(f"batch: {a.n} 条来自 {logname[:44]}   norm={'per-element' if ds.mu is not None else 'NONE'}")
    g = ds.denorm_future(batch["future"].numpy())
    print(f"真值末点距离 {np.linalg.norm(g[:, -1], axis=-1).mean():.2f} m   "
          f"真值粗糙度 {rough(g).mean():.4f} m")

    res = []
    for obj in a.objs.split(","):
        ss = {"reg": 1, "ddpm": 15, "fm": 4}[obj]
        res.append(run(obj, batch, ds, dev, steps=a.steps, lr=a.lr, sample_steps=ss,
                       basis_m=a.basis_m, trace_every=a.trace_every))
    ok = judge(res, strict=(a.n == 1))

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
