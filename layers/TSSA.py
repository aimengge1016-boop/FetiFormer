import torch
import torch.nn as nn
from einops import rearrange


class AttentionTSSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        # 兼容性保留（实际不使用）
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        # 兼容性保留
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(
        self,
        queries,
        keys=None,
        values=None,
        attn_mask=None,
        tau=None,
        delta=None,
        time_bias=None        # 🔧 MOD-1：新增 time_bias 输入
    ):
        """
        queries: [B, L, H, E]
        time_bias:
            None
            or [B, 1, 1]
            or [B, H, 1]
        """

        # [B, L, H, E] -> [B, H, L, E]
        w = rearrange(queries, 'b l h e -> b h l e')
        b, h, N, d = w.shape

        # ---------- 原始 TSSA 能量计算 ----------
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        energy = torch.sum(w_sq, dim=-1)  # [B, H, N]

        # =====================================================
        # 🔧 MOD-2：Time2Vec → TSSA Attention Bias（核心）
        # =====================================================
        if time_bias is not None:
            # time_bias 期望形状：
            # [B, 1, 1] 或 [B, H, 1]
            if time_bias.dim() == 3:
                # [B, H, 1] -> broadcast 到 [B, H, N]
                energy = energy + time_bias
            else:
                # [B, 1, 1] -> broadcast
                energy = energy + time_bias

        # ---------- TSSA 稀疏分布 ----------
        Pi = self.attend(energy * self.temp)  # [B, H, N]

        # =====================================================
        # 🔧 MOD-ATTN: 提取并返回 Query-Key 注意力矩阵（用于可视化）
        # =====================================================
        # 计算 Query-Key 相似性矩阵（归一化后的点积相似性）
        # 使用余弦相似性作为注意力权重：Q · K^T / (||Q|| ||K||)
        # 由于 TSSA 中 queries=keys，所以是自注意力
        # [B, H, N, E] -> [B, H, N, N]
        w_normalized = torch.nn.functional.normalize(w, dim=-1, p=2)  # L2归一化
        # 计算余弦相似性矩阵
        attn_matrix = torch.matmul(w_normalized, w_normalized.transpose(-1, -2))
        # 对每个head的注意力进行归一化到[0,1]
        attn_matrix = (attn_matrix + 1) / 2  # 从[-1,1]映射到[0,1]

        # ---------- 原始 TSSA 后续计算 ----------
        Pi_norm = Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)

        dots = torch.matmul(
            Pi_norm.unsqueeze(-2),
            w ** 2
        )  # [B, H, 1, E]

        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w * Pi.unsqueeze(-1), attn)

        # [B, H, L, E] -> [B, L, H, E]
        out = rearrange(out, 'b h l e -> b l h e')

        # =====================================================
        # 🔧 MOD-3：保持返回接口不变（返回注意力矩阵用于可视化）
        # =====================================================
        # 将 [B, H, N, N] 的注意力矩阵转换为 [B, N, H, N] 形状以保持一致性
        attn_reshaped = attn_matrix.transpose(1, 2)  # [B, N, H, N]

        return out, attn_reshaped
