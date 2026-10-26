"""训练循环：在真实 nuPlan 数据上训三个模型之一。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/runners/train.py \
        --objective fm --steps 4000 --batch 64 --out /home/parick/dsh_ml/runs/fm

split 语义（协议 §5）：0 = 模型训练，1 = oracle 标注 + 选择器训练，2 = 选择器测试。
本脚本只用 split 0 训练，用 split 1 做训练期验证（不碰 split 2）。
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "data"))
sys.path.insert(0, os.path.join(CODE, "models"))
sys.path.insert(0, os.path.join(CODE, "metrics"))

from dataset import MiniTrajDataset            # noqa: E402
from model import TrajectoryModel, EMA          # noqa: E402
from traj_metrics import min_ade, min_fde, miss_rate   # noqa: E402


def code_hash(root=None):
    """**整个 code/ 目录** + norm.json 的内容哈希，写进 checkpoint 与 log.json。

    ⚠ 曾经只哈希 code/models/*.py。审查指出这给了虚假的安心感：改 `train.py` 的
    评估逻辑、改 `dataset.py`、改 `traj_metrics.py`、甚至重算 `norm.json`，
    哈希都不会变，于是两个"同一 code_hash"的结果其实不可比。
    """
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(CODE):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__")))
        for f in sorted(filenames):
            if f.endswith(".py"):
                with open(os.path.join(dirpath, f), "rb") as fh:
                    h.update(os.path.relpath(os.path.join(dirpath, f), CODE).encode())
                    h.update(fh.read())
    if root:                                   # 归一化统计量也是"代码"的一部分
        p = os.path.join(root, "norm.json")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def cv_baseline(batch, scale, F=80, dt=0.1):
    """匀速外推基线（米制，K=1）。

    这是判断"模型是否正常"的**最低门槛**：目标车历史最后一个速度直接线性外推。
    历史是目标车 t0 坐标系下的 (x, y, vx, vy)，末点≈原点，故只取速度。
    任何预测模型如果 minADE 还不如它，就说明模型没在用历史信息。
    """
    h = batch["hist"].cpu().numpy() * scale                  # (B,20,4) 米 / (m/s)
    v = h[:, -1, 2:4]                                        # (B,2)
    k = np.arange(1, F + 1)[None, :, None] * dt              # (1,F,1)
    return v[:, None, None, :] * k                           # (B,1,F,2)


def roughness(traj):
    """协议 §4 的 smoothness：二阶差分幅度 |d²p|（米制，与加速度成正比）。"""
    a = traj[..., 2:, :] - 2 * traj[..., 1:-1, :] + traj[..., :-2, :]
    return np.linalg.norm(a, axis=-1)


def evaluate(model, batch, ds_val, K, sample_steps, dev, generator=None):
    """指标在**米制**下计算（数据集做了逐元素归一化，需反归一化）。"""
    model.eval()
    b = {k: v.to(dev) for k, v in batch.items()}
    preds, nfe = model.sample(b, K=K, steps=sample_steps, generator=generator)
    p = ds_val.denorm_future(preds.cpu().numpy())
    g = ds_val.denorm_future(b["future"].cpu().numpy())
    cv = cv_baseline(batch, ds_val.scale)
    return {
        "minADE": float(min_ade(p, g).mean()),
        "minFDE": float(min_fde(p, g).mean()),
        "missRate": float(miss_rate(p, g)),
        "nfe": int(nfe),
        "CV_minADE": float(min_ade(cv, g).mean()),           # 匀速外推参考线
        "CV_minFDE": float(min_fde(cv, g).mean()),
        # 粗糙度比值：>1 表示比真值更抖。轻量模型"看着正常"的硬判据。
        "rough_ratio": float(roughness(p).mean() / max(roughness(g).mean(), 1e-9)),
        "rough_gt": float(roughness(g).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["reg", "ddpm", "fm"])
    ap.add_argument("--root", default="/home/parick/dsh_ml/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2,
                    help="WSL 仅 ~2GB 可用内存，worker 过多会 OOM（每个 worker 各持一份日志缓存）")
    ap.add_argument("--cache-logs", type=int, default=1,
                    help="每个 worker 常驻的日志数；npz 单个最大 ~38MB")
    ap.add_argument("--sample-steps", type=int, default=None,
                    help="默认：fm=4, ddpm=15, reg=1")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-smooth", type=float, default=0.0,
                    help="平滑项权重（二阶差分幅度）；0 = 关闭。三个模型一致施加")
    ap.add_argument("--basis-m", type=int, default=24,
                    help="输出只允许落在前 m 个 DCT 模式（固定低通约束）；"
                         "m>=80 时退化为不做约束。实测 m=24 保留真值 99.9996%% 能量")
    ap.add_argument("--loss-power", type=float, default=0.0,
                    help="逐元素损失权重 = sd^p / mean(sd^p)。0=现状（归一化空间等权，"
                         "等于把早期步权重放大 2e6 倍）；2=米制 MSE（与 minADE 口径一致）")
    ap.add_argument("--wta-k", type=int, default=1,
                    help="best-of-K（WTA-min）：抽 K 组 (t,eps) 只回传最接近真值的那条。"
                         "治多模态坍塌，代价是前向次数 xK。1=关闭")
    a = ap.parse_args()

    arch = dict(objective=a.objective, traj_len=80, hidden_dim=192, heads=6,
                dit_depth=4, dit_mlp=2.0, enc_depth=3, drop=0.1, w_smooth=a.w_smooth,
                basis_m=a.basis_m, loss_power=a.loss_power, wta_k=a.wta_k)
    chash = code_hash(a.root)

    ss = a.sample_steps or {"reg": 1, "ddpm": 15, "fm": 4}[a.objective]
    # 回归只有一条确定性输出；K 由**模型实际返回**的候选数决定，不由调用方约定
    K = 1 if a.objective == "reg" else 6
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    ds = MiniTrajDataset(a.root, split=0, cache_logs=a.cache_logs)
    ds_val = MiniTrajDataset(a.root, split=1, cache_logs=1)
    print(f"objective={a.objective}  device={dev}  "
          f"train {len(ds):,} / val {len(ds_val):,}  batch={a.batch} steps={a.steps}")
    print(f"归一化: 逐元素 mu/sd，hist/neigh 全局 scale = {ds.scale:.4f} m  "
          f"(指标在米制下计算；三种目标的回归对象都是 x0，归一化后 var≈1)")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0)

    # 验证集必须是**跨 log 的固定种子随机子集**。
    # 审查实测：早先用 shuffle=False 取前 512 行，实为**单个 log 的 512 个连续帧
    # （仅 9 条 track、121.5 s）**，指标随取样方式漂移 12–21%，且无有效 CI。
    from torch.utils.data import Subset
    rng = np.random.default_rng(1234)
    vidx = rng.choice(len(ds_val), size=min(a.eval_n, len(ds_val)), replace=False)
    vb = next(iter(DataLoader(Subset(ds_val, vidx.tolist()), batch_size=a.eval_n,
                              shuffle=False, num_workers=0)))
    vlogs = sorted({ds_val.log_of(int(i)) for i in vidx})
    print(f"验证集: {len(vidx)} 样本, 跨 {len(vlogs)} 个 log (固定种子 1234)")

    model = TrajectoryModel(**arch, norm_mu=ds.mu, norm_sd=ds.sd).to(dev)
    pr = model.param_report()
    print(f"params: {pr}\narch: {arch}\ncode_hash: {chash}")
    print(f"输出被限制在 {pr['dof']} 维低通子空间（米制速度剖面 DCT 前 {a.basis_m} 模式）")
    print(f"损失: 逐元素权重 sd^{a.loss_power:.2f}（归一化到均值 1）  "
          f"best-of-K = {a.wta_k}")
    ema = EMA(model, decay=0.999)
    # 评估用**固定种子**的噪声发生器：否则每次 eval 重采样噪声，
    # 曲线上的抖动分不清是"训练在动"还是"采样噪声"，也无法跨步比较。
    evg = torch.Generator(device=dev)
    evg.manual_seed(999)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(i):
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        p = (i - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))

    log = {"objective": a.objective, "args": vars(a), "params": pr,
           "arch": arch, "code_hash": chash,
           "sample_steps": ss, "K_eval": K, "history": [], "train_loss": []}
    t0 = time.time()
    it = iter(dl)
    run_loss, run_n = 0.0, 0

    for i in range(1, a.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(i)

        model.train()
        b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        out = model(b)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        run_loss += float(loss.detach())
        run_n += 1

        if i % 100 == 0 or i == 1 or i == a.steps:
            print(f"  step {i:5d}/{a.steps}  loss {run_loss/max(run_n,1):.4f}  "
                  f"lr {lr_at(i):.2e}  {time.time()-t0:6.1f}s", flush=True)
            log["train_loss"].append({"step": i, "loss": run_loss / max(run_n, 1)})
            run_loss, run_n = 0.0, 0

        if i % a.eval_every == 0 or i == a.steps:
            # 同时评估 EMA 权重与原始权重。
            # 审查发现：decay=0.999 的 EMA 在 step=1000 时其实还基本是随机初始化
            # （有效窗口 1000 步），只看 EMA 会误判"模型完全没学到"。故 EMA 用偏差
            # 校正 shadow/(1−decay^t)，并同时给出 raw 权重作为对照。
            bak = copy.deepcopy(model.state_dict())
            for tag, sdict in (("ema", ema.state_dict(i)), ("raw", bak)):
                model.load_state_dict(sdict)
                m = evaluate(model, vb, ds_val, K, ss, dev, generator=evg)
                m["step"], m["weights"] = i, tag
                log["history"].append(m)
                print(f"  [eval@{i} {tag}] minADE@K {m['minADE']:.3f}  "
                      f"minFDE@K {m['minFDE']:.3f}  miss {m['missRate']:.3f}  "
                      f"nfe {m['nfe']}  粗糙度 {m['rough_ratio']:6.2f}x  "
                      f"| 匀速外推 minADE {m['CV_minADE']:.3f} "
                      f"({'模型更好' if m['minADE'] < m['CV_minADE'] else '不如基线!'})",
                      flush=True)
            model.load_state_dict(bak)
            del bak

    ckpt = os.path.join(a.out, "model.pt")
    torch.save({
        "state_dict": ema.state_dict(a.steps),
        "objective": a.objective,
        "arch": arch,                 # 构造参数：缺了它权重无法复现
        "code_hash": chash,           # code/models/*.py 的内容哈希
        "args": vars(a),
        "sample_steps": ss,
    }, ckpt)
    with open(os.path.join(a.out, "log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved {ckpt}  ({os.path.getsize(ckpt)//1024} KB)   "
          f"总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
