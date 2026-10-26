"""基础层：自包含的 Mlp / DropPath / 时间步嵌入。

刻意不依赖 timm —— Diffusion-Planner 用 `timm.models.layers.Mlp`，
我们只需要它几行的功能，自己写可以让代码自包含、也更好讲清楚。

（文件名曾是 nn.py，会与 torch.nn 的惯用别名 nn 混淆，故改为 layers.py。）
"""
import math

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DropPath(nn.Module):
    """随机深度。eval 或 drop_prob==0 时退化为恒等。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def timestep_embedding(t, dim, max_period=10000):
    """正弦时间步嵌入。t: (B,) -> (B, dim)。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
