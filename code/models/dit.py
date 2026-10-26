"""DiT 去噪骨干 —— 只借鉴 Diffusion-Planner 的**关键思路**，保持极轻量。

对照 `Diffusion-Planner/.../model/module/{dit,decoder}.py`，真正起作用的是三点：

1. **整条轨迹 = 一个 token**（他们的 `preproj: output_dim → hidden`）。
   早先把 80 个时间步拆成 80 个 token 各自输出，等于给每步独立注入噪声，
   轨迹就**坑坑洼洼**。压成单个 token 后 160 个输出由同一个线性层产生，
   相邻时间步天然连续 —— 这是最要紧的平滑先验，而且是**零成本**的。

2. **cross-attention 到场景 token**，让历史/邻居**直连**进解码器。
   早先只有 adaLN（把整个场景压成一个加法向量），拐弯这类信息几乎传不进去。

3. **预测 x0（干净轨迹），不预测 ε**（见 model.py）。他们默认 `model_type="x_start"`。

**刻意省略**（他们有大算力我们没有）：
- 自注意力：本模型 P=1（只生成目标车一条轨迹），对单 token 做自注意力是恒等操作，白占 4D² 参数。
- 路线编码 RouteEncoder、邻居联合预测、guidance：留作后续扩展。
"""
import torch
import torch.nn as nn

from layers import Mlp, TimestepEmbedder, modulate


class DiTBlock(nn.Module):
    """cross-attention 到场景 token + adaLN 调制 MLP。P=1 故无需自注意力。"""

    def __init__(self, dim=192, heads=6, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x, cross_c, y, cross_mask=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN_modulation(y).chunk(6, dim=1)
        h = modulate(self.norm1(x), sh_a, sc_a)
        x = x + self.cross_attn(h, cross_c, cross_c,
                                key_padding_mask=cross_mask, need_weights=False)[0]
        h = modulate(self.norm2(x), sh_m, sc_m)
        x = x + g_m.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim=192, out_dim=160):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = Mlp(dim, dim * 2, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        return self.proj(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """x_flat: (B, traj_len*2)；t: (B,)；tokens: (B,N,D) 场景 token。"""

    def __init__(self, traj_len=80, dim=192, depth=4, heads=6, mlp_ratio=2.0,
                 drop=0.1, pre_hidden=256):
        super().__init__()
        self.traj_len = traj_len
        # +1 是**锚定槽位**：轨迹第 0 点 = 当前状态（本任务坐标系下恒为原点）。
        # 审查实测：不锚定时流匹配首个预测点误差 0.91 m，而真值离原点仅 0.50 m。
        # 参考实现同样前置当前状态并在**每个采样步**重新钉住
        # （decoder.py:120 prepend, decoder.py:136 correcting_xt_fn）。
        self.out_len = traj_len + 1
        self.out_dim = self.out_len * 2
        self.preproj = Mlp(self.out_dim, pre_hidden, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio, drop) for _ in range(depth)]
        )
        self.final = FinalLayer(dim, self.out_dim)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(basic)
        # adaLN-Zero：调制层与输出层零初始化 → 训练初期接近恒等，避免一开始就喷射
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.proj.fc2.weight, 0)
        nn.init.constant_(self.final.proj.fc2.bias, 0)

    def forward(self, x_flat, t, tokens, cross_mask=None):
        # 池化摘要进 adaLN。**必须带掩码**：审查实测未掩码时 padding 槽位的 token
        # 并非零向量（‖13.86‖ = 有效 token），使条件向量偏移 ‖Δy‖/‖y‖ ≈ 11%，
        # 有效邻居越少污染越重（仅 1 个邻居时达 20%）。
        if cross_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            w = (~cross_mask).float().unsqueeze(-1)              # (B, N, 1)
            pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1.0)
        y = self.t_embedder(t) + pooled
        h = self.preproj(x_flat).unsqueeze(1)                    # (B, 1, D)
        for blk in self.blocks:
            h = blk(h, tokens, y, cross_mask)
        return self.final(h, y).squeeze(1)
