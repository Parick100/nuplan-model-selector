"""共享上下文编码器。

结构对照 Diffusion-Planner 的 `model/module/{mixer,encoder}.py`，但只保留
本项目需要的部分（agent-only，地图留作 Phase 2 的消融项）：

    hist  (B, T, 4)           目标车历史
    neigh (B, M, T, 6)        邻居历史 + 尺寸
    neigh_mask (B, M)         有效槽位

    -> tokens (B, 1+M, D)     每个智能体一个 token（自车在最前）
    -> context (B, D)         自车 token，供解码器条件化

**公平性关键**：三个模型（回归 / 扩散 / 流匹配）共用同一个编码器实例定义，
只有解码器（"想象"那一步）不同。
"""
import torch
import torch.nn as nn

from layers import Mlp


class MixerBlock(nn.Module):
    """沿时间轴与通道轴各做一次 MLP（MLP-Mixer 的核心思想）。"""

    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(tokens_mlp_dim, tokens_mlp_dim, tokens_mlp_dim, drop=drop)
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(channels_mlp_dim, channels_mlp_dim, channels_mlp_dim, drop=drop)

    def forward(self, x):
        # x: (..., T, C)
        y = self.norm1(x).permute(0, 2, 1)
        y = self.tokens_mlp(y).permute(0, 2, 1)
        x = x + y
        return x + self.channels_mlp(self.norm2(x))


class AgentHistoryEncoder(nn.Module):
    """把 (B, A, T, F) 的每个智能体历史压成一个 D 维 token。"""

    def __init__(self, time_len, feat_dim, hidden_dim=192, channels=128, tokens_mlp=64,
                 depth=2, drop=0.1):
        super().__init__()
        self.channel_pre = Mlp(feat_dim + 1, channels, channels)   # +1 是有效掩码
        self.token_pre = Mlp(time_len, tokens_mlp, tokens_mlp)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp, channels, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(channels)
        self.emb = Mlp(channels, hidden_dim, hidden_dim)

    def forward(self, x, valid):
        """x: (B, A, T, F)  valid: (B, A) bool"""
        B, A, T, F = x.shape
        vf = valid.float().unsqueeze(-1).unsqueeze(-1).expand(B, A, T, 1)
        x = torch.cat([x, vf], dim=-1).reshape(B * A, T, F + 1)
        x = self.channel_pre(x).permute(0, 2, 1)          # (BA, C, T)
        x = self.token_pre(x).permute(0, 2, 1)            # (BA, T, C)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)                                  # 时间维池化
        return self.emb(self.norm(x)).view(B, A, -1)


class ContextEncoder(nn.Module):
    def __init__(self, time_len=20, hidden_dim=192, heads=6, depth=3, drop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 自车：4 维（x,y,vx,vy）；邻居：6 维（+ w,l）
        self.target_enc = AgentHistoryEncoder(time_len, 4, hidden_dim)
        self.neigh_enc = AgentHistoryEncoder(time_len, 6, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=drop, batch_first=True, norm_first=True, activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hist, neigh, neigh_mask):
        B = hist.shape[0]
        # 自车补一个"智能体"维度，与邻居共用同一套编码器实现
        tgt = self.target_enc(hist.unsqueeze(1),
                              torch.ones(B, 1, dtype=torch.bool, device=hist.device))
        nb = self.neigh_enc(neigh, neigh_mask)
        tokens = torch.cat([tgt, nb], dim=1)                       # (B, 1+M, D)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=hist.device),
                         ~neigh_mask.bool()], dim=1)               # True = 忽略
        tokens = self.fusion(tokens, src_key_padding_mask=pad)
        tokens = self.norm(tokens)
        return tokens, tokens[:, 0]                                # 自车 token 作 context
