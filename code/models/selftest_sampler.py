"""采样器自检与步数扫描 —— 把"采样器代数"与"模型学得好不好"分开。

**核心思想**：构造一个**神谕模型**，它的 x̂0 在任何 t 上都是精确的 x0。若采样器
代数正确，从 x_1 = ε 出发应当精确收敛到 x0，误差随步数单调下降到浮点精度。
这样就把两个完全不同的失败模式分开了：

  · 神谕下不收敛  → 采样器 / 参数化 / 时间方向的 bug（本文件的 A 部分）
  · 神谕下收敛、真模型不收敛 → 训练问题，不是采样器问题（本文件的 B 部分）

两种范式统一用 x0 参数化的 DDIM 更新，所以两条路径都要测。

    /home/parick/ENTER/envs/diffusion_planner/bin/python code/models/selftest_sampler.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from model import TrajectoryModel, alpha_bar                    # noqa: E402

B, T = 8, 80


def oracle_sample(obj, x0, steps, dev):
    """用"x̂0 ≡ x0"的神谕场跑完整的采样更新，返回终端 x。"""
    x = torch.randn(B, T, 2, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        out = x0                                       # ← 神谕
        if obj == "ddpm":
            ab, abn = alpha_bar(t), alpha_bar(tn)
            eps = (x - torch.sqrt(ab) * out) / torch.sqrt(1 - ab)
            x = torch.sqrt(abn) * out + torch.sqrt(1 - abn) * eps
        else:
            eps = (x - (1 - t) * out) / t
            x = (1 - tn) * out + tn * eps
    return x


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 用一条真实量级的归一化轨迹（均值 0、逐元素方差 ≈1）当 x0
    x0 = torch.randn(B, T, 2, device=dev) * 1.0

    print("=== A) 神谕采样器：从 ε 出发能否精确回到 x0 ===")
    print(f"{'obj':>6} {'steps':>6} {'rel err':>12} {'max|Δ|':>12}")
    bad = []
    for obj in ("fm", "ddpm"):
        prev = None
        for steps in (1, 2, 4, 8, 15, 32, 64, 128, 256):
            torch.manual_seed(0)
            y = oracle_sample(obj, x0, steps, dev)
            rel = float((y - x0).norm() / x0.norm())
            mx = float((y - x0).abs().max())
            print(f"{obj:>6} {steps:>6} {rel:>12.3e} {mx:>12.3e}")
            prev = rel
        if prev > 1e-4:
            bad.append((obj, prev))

    print("\n判读：相对误差应随步数下降。fm 的线性路径在 eps 恒定时是精确的")
    print("      （x_t' = (1−t')x0 + t'·(x_t−(1−t)x0)/t，若 x_t=(1−t)x0+tε 则恒等），")
    print("      故 fm 应当在 1 步内就到机器精度；ddpm 的 VP 路径在 x̂0=x0 时同样精确。")

    print("\n=== B) 真实模型：步数扫描（锯齿是离散化还是场本身）===")
    root = "/home/parick/dsh_ml/runs"
    want = sys.argv[1:] or ["fm", "ddpm"]
    for obj in want:
        p = os.path.join(root, obj, "model.pt")
        if not os.path.exists(p):
            print(f"  {obj}: 无 checkpoint，跳过")
            continue
        ck = torch.load(p, map_location=dev)
        m = TrajectoryModel(**ck["arch"]).to(dev)
        try:
            m.load_state_dict(ck["state_dict"])
        except RuntimeError as e:
            print(f"  {obj}: 架构不符，跳过 ({str(e).splitlines()[0][:60]})")
            continue
        m.eval()
        batch = {"hist": torch.randn(B, 20, 4, device=dev),
                 "neigh": torch.randn(B, 10, 20, 6, device=dev),
                 "neigh_mask": torch.rand(B, 10, device=dev) > 0.3,
                 "future": x0}
        print(f"\n  --- {obj}  code_hash={ck.get('code_hash','?')} "
              f"steps_trained={ck.get('args',{}).get('steps','?')} ---")
        with torch.no_grad():
            tokens, mask, _ = m.encode(batch)
            # t=1 处：模型是否在照抄输入噪声？
            x1 = torch.randn(B, T, 2, device=dev)
            t1 = torch.ones(B, device=dev)
            o1 = m.predict_field(torch.cat(
                [torch.zeros(B, 1, 2, device=dev), x1], dim=1).reshape(B, -1),
                t1, tokens, mask).reshape(B, m.out_len, 2)[:, 1:]
            cc = float(torch.corrcoef(torch.stack([o1.reshape(-1), x1.reshape(-1)]))[0, 1])
            print(f"    t=1: ‖x̂0‖={float(o1.norm()/B**.5):.3f}  ‖x_t‖={float(x1.norm()/B**.5):.3f}"
                  f"  corr(x̂0, x_t)={cc:+.4f}")
            print(f"    → corr 应接近 0（模型应忽略纯噪声）；若接近 1 说明在照抄输入")
            for steps in (1, 2, 4, 8, 16, 32):
                pr, nfe = m.sample(batch, K=1, steps=steps)
                print(f"    steps {steps:>3}  nfe {nfe:>4}  "
                      f"‖pred−x0‖ {float((pr[:,0]-x0).norm()/x0.norm()):.4f}")


if __name__ == "__main__":
    main()
