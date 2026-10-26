"""统一模型：共享编码器 + **共享解码器** + 三种目标（回归 / 扩散 / 流匹配）。

借鉴 Diffusion-Planner 的关键决定（详见 dit.py 与各段注释）：

1. **整条轨迹 = 一个 token**（`preproj`）—— 相邻时间步天然连续，零成本平滑先验。
2. **cross-attention 到场景 token** —— 历史/邻居直连进解码器。
3. **三种目标统一预测 x0（干净轨迹）而非 ε / v**（他们默认 `model_type="x_start"`）。
   审查用 λ 混合实验证明：即使场误差仅 1e-4，ε 参数化仍比流匹配差 7 倍，
   因为 x̂0 = (x_t−√(1−ᾱ)ε̂)/√ᾱ 会把场误差放大 √((1−ᾱ)/ᾱ)（t=1 处达 152 倍）。
   统一后"扩散 vs 流匹配"的差别**只剩路径本身**。
4. **锚定当前状态**：轨迹含第 0 点（本坐标系下恒为原点），训练不注入噪声、
   采样每一步重新钉住。
5. **固定 DCT 低通约束，且作用在「速度剖面」上**（见 `velocity_projector`）：
   输出头 162 个值来自同一个线性层，相邻时间步之间没有任何结构 —— 这是
   "轨迹看着像噪声"的**结构性**原因，光加平滑损失压不住。
6. **回归也走同一个 DiT**（输入置零、t=0）。审查指出：早先回归用 284 K 参数的
   MLP 头、生成用 2.47 M 的 DiT，等于把"生成范式更好"与"解码器更大 + 有邻居
   交叉注意力"混在一起，而且混淆方向是**偏向生成臂**的。现在三条臂参数量
   完全相同（encoder 1,228,864 + DiT 2,465,890 = 3,694,754），只差生成机制。

**极轻量**：编码器 1.23 M + DiT 2.47 M = 3.69 M（参考实现远大于此）。
"""
import copy
import math

import torch
import torch.nn as nn

from dit import DiT
from encoder import ContextEncoder


def dct_lowpass(n, m):
    """前 m 个 DCT-II 模式的正交投影矩阵 (n, n)。（位置空间；保留供诊断使用）"""
    k = torch.arange(m, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    B = torch.cos(math.pi * (i + 0.5) * k / n)
    B[0] *= 1.0 / math.sqrt(2.0)
    B = B / B.norm(dim=1, keepdim=True)
    return B.t() @ B


def velocity_projector(n, m):
    """约束**速度剖面**的低通算子 P_v = C · (BᵀB) · A，形状 (n, n)。

    为什么不是直接对位置做低通（曾经的实现，`dct_lowpass`）：
    DCT-II 在两端做**偶延拓**，而"从原点出发、速度恒定的轨迹"在端点处**导数不连续**
    （偶延拓后是折线），截断到前 m 个模式就产生 Gibbs 式误差。实测（2000 条真值，
    M=24，`code/runners/diag_boundary.py`）：

        · 匀速轨迹的端点偏差 = **0.2744·(v·dt)**（15 m/s 时 0.41 m），两端**同号**；
        · 真实真值上，残差的**有符号偏差**首点 0.146 / 末点 0.149 m，
          占总残差模长的 **56.6%** —— 是系统性偏置，不是噪声；
        · minFDE 地板 0.1706 m，而"99.9996% 能量保留"这个平均统计**完全看不见**它。

    改成约束速度后用同一个序列（81 点 = [锚点 0, p₀…p₇₉]，80 个一阶差分）：

        A (n×n)  p ↦ Δ，Δ₀ = p₀ − 0，Δᵢ = pᵢ − pᵢ₋₁      （单位下三角 + 负次对角）
        BᵀB      把 Δ 投影到前 m 个 DCT 模式
        C (n×n)  Δ ↦ 累积和（从锚点的 0 出发）            （全 1 下三角）

    因为 C·A = I（差分再累积和是恒等），P_v² = C·(BᵀB)·(A·C)·(BᵀB)·A = P_v，
    仍是幂等投影，trace = m。而**匀速轨迹的 Δ 是常向量**，全部落在第 0 个模式上，
    所以 P_v 对匀速轨迹**精确恒等**。实测改善（同上脚本）：

        · 匀速端点偏差 0.2744·(v·dt) → **0**
        · 有符号偏差 0.1485 → **0.0027 m**（−98.2%），偏置占比 56.6% → **3.4%**
        · minFDE 地板 0.1706 → **0.0000 m**，minADE 地板 0.0426 → 0.0281 m

    ⚠ 实现时**必须把锚点纳入差分**。第一次写的时候只在 80 点未来上做 L·P·L⁻¹，
    于是 L·A = I − p₀·1 不是恒等：算子既不幂等（|P²−P| = 0.61），
    又在末点产生整整 1.0·(v·dt) 的偏移。"积分回去"必须从**已知的锚点**出发。
    """
    assert m < n, "m >= n 时无需约束"
    Pv = dct_lowpass(n, m)
    A = torch.eye(n) - torch.diag(torch.ones(n - 1), -1)
    C = torch.tril(torch.ones(n, n))
    return C @ Pv @ A


BETA_MIN, BETA_MAX = 0.1, 20.0


def alpha_bar(t, bmin=BETA_MIN, bmax=BETA_MAX):
    """VP-SDE 的 ᾱ(t)，与 Diffusion-Planner `sde.py` 一致（已符号+数值核对）。"""
    return torch.exp(-0.5 * t ** 2 * (bmax - bmin) - bmin * t)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)                      # buffer（mu/sd/proj）直接同步，不参与平均

    @torch.no_grad()
    def state_dict(self, step=None):
        """EMA 权重。

        ⚠ **不要做 Adam 式的偏差校正**（`shadow/(1−decay^t)`）。这里 shadow **初始化
        于 θ₀ 而不是 0**，所以
            shadow_t = decay^t·θ₀ + (1−decay^t)·θ̄_t
        已经是 θ₀ 与 θ̄ 的**凸组合**，本来就无缩放偏差；再除以 (1−decay^t) 会把它放大。
        实测（θ₀=1 → θ=10，由独立审查用本类跑出）：t=1 时 shadow=1.009，
        "校正"后 **1009.0**；t=1000 时 shadow=6.691，校正后 10.58 —— 数字看着对了，
        代价是把 1.58 倍的随机初始化一起放大，且 step=1000 的 eval 会虚高 1.58 倍。
        那种校正只适用于 shadow 从 0 起步的动量缓冲（Adam 的 m/v）。

        真正的问题是**陈旧**而非偏差：decay=0.999 在 step 1000 时仍有 37% 的随机初始化
        权重。正确应对是**同时报原始权重**（train.py 的 ema/raw 双评估），
        到 step 20000 时 decay^t = 2e-9，EMA 已完全跟上。
        """
        return {k: (v.clone() if v.is_floating_point() else v)
                for k, v in self.shadow.state_dict().items()}


def smoothness_loss(traj_m):
    """二阶差分幅度（与加速度成正比）。traj_m: (..., T, 2)，**米制**。

    ⚠ 必须传米制。曾经传归一化轨迹，于是相对权重变成 1/sd²，
    而 sd 跨度 0.023→33 m（权重差 2e6 倍）：惩罚几乎全落在早期步（那里任何归一化
    曲率在米制下都可忽略），对**后期**步的可见抖动几乎无感。这正是 `--w-smooth 10`
    仍然让输出比真值粗糙 84 倍的原因。
    """
    acc = traj_m[..., 2:, :] - 2 * traj_m[..., 1:-1, :] + traj_m[..., :-2, :]
    return (acc ** 2).sum(dim=-1).mean()


class TrajectoryModel(nn.Module):
    def __init__(self, objective="fm", traj_len=80, hidden_dim=192, heads=6,
                 dit_depth=4, dit_mlp=2.0, enc_depth=3, drop=0.1, w_smooth=0.0,
                 basis_m=24, norm_mu=None, norm_sd=None, loss_power=0.0, wta_k=1):
        super().__init__()
        assert objective in ("reg", "ddpm", "fm")
        self.objective = objective
        self.traj_len = traj_len
        self.out_len = traj_len + 1                 # 含锚定槽位
        self.hidden_dim = hidden_dim
        self.w_smooth = float(w_smooth)
        self.basis_m = int(basis_m)
        self.loss_power = float(loss_power)
        self.wta_k = int(wta_k)

        self.encoder = ContextEncoder(time_len=20, hidden_dim=hidden_dim,
                                      heads=heads, depth=enc_depth, drop=drop)
        # 三条臂共用同一个 DiT —— 参数量严格相同，差别只在生成机制
        self.dit = DiT(traj_len=traj_len, dim=hidden_dim, depth=dit_depth,
                       heads=heads, mlp_ratio=dit_mlp, drop=drop)

        self.register_buffer(
            "proj",
            velocity_projector(traj_len, self.basis_m) if self.basis_m < traj_len
            else torch.eye(traj_len))
        self.set_norm(norm_mu, norm_sd)

    # ------------------------------------------------- 平滑子空间约束
    def set_norm(self, mu, sd):
        """把低通约束从**归一化空间**搬到**米制空间**。

        归一化是逐时间步除以 sd（sd 从 0.023 m 长到 33 m），所以对归一化轨迹做低通
        ≠ 对米制轨迹做低通：早期步被放大、后期步被压缩，而后期恰恰是毛刺最明显的
        地方。故先反归一化 → 投影 → 再归一化：

            p̃ ← ( P_v(p̃·sd + mu) − mu ) / sd

        这是一个仿射算子 p̃ ↦ A p̃ + b，b = (P_v mu − mu)/sd 是常数。
        A² = A 且 A b = 0，所以它**真的**是把输出限制在"米制下速度剖面只含前
        basis_m 个 DCT 模式"这个 ≤2·basis_m 维仿射子空间里。

        ⚠ 必须**逐坐标**沿时间轴投影（einsum）。曾经把 (B,80,2) 直接 reshape 成
        160 维再乘 block-diag 投影矩阵 —— 那样 160 维的排布是 x0,y0,x1,y1,…
        （交替），与 block-diag 假设的 (x0..x79,y0..y79) 不符，等于把 x/y 交错当成
        一条一维信号去低通，直接毁掉 49% 能量、米制残差 19.4 m。

        ⚠ mu/sd 必须 persistent：否则加载 checkpoint 时归一化会静默退回恒等。
        """
        n = self.traj_len
        self.register_buffer("mu", torch.zeros(n, 2) if mu is None else
                             torch.as_tensor(mu, dtype=torch.float32).reshape(n, 2))
        self.register_buffer("sd", torch.ones(n, 2) if sd is None else
                             torch.as_tensor(sd, dtype=torch.float32).reshape(n, 2))

    def _proj_fut(self, fut):
        """fut (B, n, 2) 归一化 -> 米制下满足速度低通约束的轨迹（仍归一化输出）。"""
        z = torch.einsum("ij,bjc->bic", self.proj, fut * self.sd + self.mu)
        return (z - self.mu) / self.sd

    def _to_metres(self, fut):
        """(B, n, 2) 归一化 -> 米制（仅供 smoothness_loss 等报告用途）。"""
        return fut * self.sd + self.mu

    @property
    def output_dim(self):
        """本模型在米制下**真正**能表达的轨迹自由度。"""
        return 2 * (self.basis_m if self.basis_m < self.traj_len else self.traj_len)

    # ------------------------------------------------------------ 内部
    def encode(self, batch):
        tokens, ctx = self.encoder(batch["hist"], batch["neigh"], batch["neigh_mask"])
        B = batch["hist"].shape[0]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
                          ~batch["neigh_mask"].bool()], dim=1)      # True = 忽略
        return tokens, mask, ctx

    def predict_field(self, x_flat, t, tokens, mask):
        """统一输出 **x̂0**（干净轨迹），并施加固定低通约束。

        回归可以看成两种范式在 **t=0、x=0** 处的退化角：此时 x_t 不含任何关于 x0 的
        信息，模型必须完全依赖上下文 —— 这正是"确定性回归"的定义。所以三条臂
        共用同一个 `predict_field`，区别只在训练目标与采样循环。
        """
        out = self.dit(x_flat, t, tokens, mask)
        out = out.reshape(-1, self.out_len, 2)
        fut = self._proj_fut(out[:, 1:, :])
        return torch.cat([out[:, :1, :], fut], dim=1).reshape(-1, self.out_len * 2)

    @staticmethod
    def _assemble(fut, B, dev):
        """把未来 (B,80,2) 拼上前置锚点，(B,81,2) -> 展平。"""
        anchor = torch.zeros(B, 1, 2, device=dev)
        return torch.cat([anchor, fut.reshape(B, -1, 2)], dim=1).reshape(B, -1)

    # ------------------------------------------------------- 损失权重
    def _elem_weight(self, dev):
        """逐元素损失权重 (out_len*2,)，作用于归一化残差上。

        **为什么需要它**：归一化把每个元素变成近似单位方差，于是归一化 MSE
        `mean((p̃−g̃)²)` 对每个元素等权。但一个元素上的米制误差是
        `(p̃−g̃)·sd`，而 sd 从 0.023 m 长到 33 m —— 也就是说**归一化 MSE 把早期步
        的权重放大了约 2e6 倍**。后果：模型在早期步精确到毫米，后期步却可以差几米，
        而 minADE/minFDE 是在**米制**下按步等权平均的。实测正是这个症状：
        回归的 `mean(Δx) = −0.39 m`（系统性欠冲）、minFDE 8.7 m。

        令 `w = sd^p / mean(sd^p)`（归一化到均值 1，保证损失尺度与 lr 不被改变）：
          p = 0 → 现状（归一化空间等权）
          p = 1 → 按米制误差的"相对尺度"加权
          p = 2 → 等价于**米制 MSE**，与 minADE 的加权口径一致

        权重只作用在未来 80 步上；锚点槽位权重固定为 1（它恒为 0，不参与学习）。
        """
        w = self.sd.reshape(-1).to(dev).clamp_min(1e-6) ** self.loss_power
        w = w / w.mean()
        return torch.cat([torch.ones(2, device=dev), w], dim=0)

    # ------------------------------------------------------------ 前向
    def forward(self, batch):
        tokens, mask, ctx = self.encode(batch)
        B = batch["future"].shape[0]

        # **目标也投影**：不然模型被要求去拟合它无法表示的成分，损失里留下一个
        # 常数地板（实测占损失方差 0.18%，但 98.2% 的残差 SSE 挤在前 10 步，
        # 梯度分布很坏）。投影目标后地板恰为 0，模型只需学"平滑后的未来"。
        # 评测仍对**原始**真值算 minADE/minFDE，所以地板如实体现为 0.028 m。
        target = self._proj_fut(batch["future"])
        gt = self._assemble(target, B, batch["future"].device)
        w = self._elem_weight(gt.device)
        K = max(1, self.wta_k)

        if self.objective == "reg":
            x0 = torch.zeros(B, self.out_len * 2, device=gt.device)
            out = self.predict_field(x0, torch.zeros(B, device=gt.device), tokens, mask)
            loss = (w * (out - gt) ** 2).mean()
        else:
            # best-of-K（WTA-min）：一次抽 K 组 (t, ε)，只对**最接近真值的那条**
            # 回传梯度。这是多模态预测里治"模式坍塌"的标准做法，而且它优化的正是
            # minADE@K 这个指标本身。代价是前向次数 ×K。
            # 背景：普通 MSE 的最优解是**条件均值**，K 条候选会全部挤向均值
            # （实测横向跨度只有 2.9–9.0 m，而真值横移 p95 是 9.5–33 m）。
            if K > 1:
                tokK = tokens.repeat_interleave(K, dim=0)
                mskK = mask.repeat_interleave(K, dim=0)
                gtK = gt.repeat_interleave(K, dim=0)
                tK = torch.rand(B * K, device=gt.device)
                noise = torch.randn_like(gtK)
                if self.objective == "ddpm":
                    ab = alpha_bar(tK).view(B * K, 1)
                    x_t = torch.sqrt(ab) * gtK + torch.sqrt(1 - ab) * noise
                else:
                    tv = tK.view(B * K, 1)
                    x_t = (1 - tv) * gtK + tv * noise
                x_t = x_t.reshape(B * K, self.out_len, 2)
                x_t[:, 0, :] = 0.0
                out = self.predict_field(x_t.reshape(B * K, -1), tK, tokK, mskK)
                per = (w * (out - gtK) ** 2).mean(dim=1).reshape(B, K)
                loss = per.min(dim=1).values.mean()
            else:
                t = torch.rand(B, device=gt.device)
                noise = torch.randn_like(gt)
                if self.objective == "ddpm":
                    ab = alpha_bar(t).view(B, 1)
                    x_t = torch.sqrt(ab) * gt + torch.sqrt(1 - ab) * noise
                else:
                    tv = t.view(B, 1)
                    x_t = (1 - tv) * gt + tv * noise
                x_t = x_t.reshape(B, self.out_len, 2)
                x_t[:, 0, :] = 0.0                 # 锚点不注入噪声
                out = self.predict_field(x_t.reshape(B, -1), t, tokens, mask)
                loss = (w * (out - gt) ** 2).mean()

        if self.w_smooth > 0:
            loss = loss + self.w_smooth * smoothness_loss(
                self._to_metres(out.reshape(-1, self.out_len, 2)[:, 1:]))
        return {"loss": loss, "x0": out}

    # ------------------------------------------------------------ 采样
    @torch.no_grad()
    def sample(self, batch, K=1, steps=None, generator=None):
        """返回 (未来 (B,K,80,2) 归一化空间, nfe)。锚点每步重钉。

        回归**不看 K**：它只有一条确定性输出，返回值 K 维恒为 1，nfe 恒为 1。
        调用方应使用 `preds.shape[1]` 而不是自己传进来的 K（旧版会在日志里
        记下"请求的 K"，与实际不符）。
        """
        tokens, mask, ctx = self.encode(batch)
        B = ctx.shape[0]

        if self.objective == "reg":
            x0 = torch.zeros(B, self.out_len * 2, device=ctx.device)
            out = self.predict_field(x0, torch.zeros(B, device=ctx.device), tokens, mask)
            fut = out.reshape(B, self.out_len, 2)[:, 1:]
            return fut.unsqueeze(1), 1

        steps = steps or {"ddpm": 15, "fm": 4}[self.objective]
        C = B * K
        tok = tokens.repeat_interleave(K, dim=0)
        msk = mask.repeat_interleave(K, dim=0)
        x = torch.randn(C, self.out_len * 2, device=ctx.device, generator=generator)
        x = x.reshape(C, self.out_len, 2)
        x[:, 0, :] = 0.0
        ts = torch.linspace(1.0, 0.0, steps + 1, device=ctx.device)

        # 两种范式统一用 **x0 参数化的 DDIM 更新**：已知 x_t 与 x̂0 反推噪声项，
        # 再用同一路径公式跳到 x_{t'}。审查已验证：神谕 x̂0 下两种范式在
        # steps ∈ {1,2,4,8,15,50,250} 上误差恒为 0.000e+00，且
        #   fm  x' = (t'/t)·x + ((t−t')/t)·out   （系数 t'/t < 1，压缩）
        #   ddpm x' = r·x + (a'−r·a)·out, r = √((1−ᾱ')/(1−ᾱ)) ≤ 1（末步 r 恰为 0）
        # 最小 t = 1/steps > 0，故 /t 与 /√(1−ᾱ) 永不除零。
        for i in range(steps):
            t, tn = ts[i], ts[i + 1]
            out = self.predict_field(x.reshape(C, -1), t.expand(C), tok, msk)
            out = out.reshape(C, self.out_len, 2)   # 约束已在 predict_field 内施加
            out[:, 0, :] = 0.0
            if self.objective == "ddpm":
                ab, abn = alpha_bar(t), alpha_bar(tn)
                eps_hat = (x - torch.sqrt(ab) * out) / torch.sqrt(1 - ab)
                x = torch.sqrt(abn) * out + torch.sqrt(1 - abn) * eps_hat
            else:
                eps_hat = (x - (1 - t) * out) / t
                x = (1 - tn) * out + tn * eps_hat
            x[:, 0, :] = 0.0                      # ← 每个采样步重新钉住锚点
        return x[:, 1:, :].reshape(B, K, self.traj_len, 2), K * steps

    # ------------------------------------------------------------ 报告
    def param_report(self):
        parts = {"encoder": count_params(self.encoder), "dit": count_params(self.dit)}
        parts["total"] = count_params(self)
        parts["dof"] = self.output_dim
        return parts


if __name__ == "__main__":
    B = 4
    batch = {
        "hist": torch.randn(B, 20, 4),
        "neigh": torch.randn(B, 10, 20, 6),
        "neigh_mask": torch.rand(B, 10) > 0.3,
        "future": torch.randn(B, 80, 2),
    }
    mu = torch.randn(80, 2).cumsum(0) * 1.5          # 假装是真值的均值曲线
    sd = torch.linspace(0.3, 6.0, 80)[:, None].expand(80, 2).contiguous()
    ref = None
    for obj in ("reg", "ddpm", "fm"):
        for lp, wk in ((0.0, 1), (2.0, 1), (2.0, 4)):
            m = TrajectoryModel(objective=obj, norm_mu=mu, norm_sd=sd,
                                loss_power=lp, wta_k=wk)
            pr = m.param_report()
            o = m(batch)
            p, nfe = m.sample(batch, K=3)
            z = torch.randn(64, 80, 2)
            idem = float((m._proj_fut(m._proj_fut(z)) - m._proj_fut(z)).abs().max())
            cv = (10.0 * 0.1 * torch.arange(1, 81))[:, None].expand(80, 2).clone()
            cvm = (cv - mu) / sd
            cv_err = float((m._proj_fut(cvm.unsqueeze(0))[0] - cvm).abs().max())
            if ref is None:
                ref = pr["total"]
            w = m._elem_weight(torch.device("cpu"))
            print(f"{obj:5s} lp={lp:.1f} K={wk} loss {o['loss'].item():.4f}  "
                  f"sample {tuple(p.shape)}  nfe {nfe}  |A²−A| {idem:.2e}  "
                  f"匀速残差 {cv_err:.2e} m  w: [{float(w.min()):.2e},{float(w.max()):.2e}]  "
                  f"{'参数一致 OK' if pr['total'] == ref else '!! 参数量不一致'}")
