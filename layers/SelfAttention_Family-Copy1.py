import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


# ======================================================
# FlowAttention
# ======================================================
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    # >>> MOD: 增加 time_bias=None（接口兼容）
    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None, time_bias=None):

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)

        normalizer_row = 1.0 / (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6)
        )
        normalizer_col = 1.0 / (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6)
        )

        normalizer_row_refine = torch.einsum(
            "nhld,nhd->nhl",
            queries + 1e-6,
            (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6
        )
        normalizer_col_refine = torch.einsum(
            "nhsd,nhd->nhs",
            keys + 1e-6,
            (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6
        )

        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (queries.shape[2] / keys.shape[2])
        )
        normalizer_col_refine = torch.softmax(
            normalizer_col_refine, dim=-1
        ) * keys.shape[2]

        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (
            (queries @ kv)
            * normalizer_row[:, :, :, None]
            * normalizer_row_refine[:, :, :, None]
        ).transpose(1, 2).contiguous()

        return x, None


# ======================================================
# FlashAttention
# ======================================================
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10
        EPSILON = 1e-10

        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1], device=Q.device)[..., None]
        m = torch.ones(Q.shape[:-1], device=Q.device)[..., None] * NEG_INF

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j, (Kj, Vj) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
            maskj = mask_BLOCKS[j] if mask is not None else None
            for i, Qi in enumerate(Q_BLOCKS):
                Oi, li, mi = O_BLOCKS[i], l_BLOCKS[i], m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale
                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)

                if maskj is not None:
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block)

                if maskj is not None:
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block - mi_new) * l_block

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                    torch.exp(m_block - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        return O, None

    # >>> MOD
    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None, time_bias=None):

        res = self.flash_attention_forward(
            queries.permute(0, 2, 1, 3),
            keys.permute(0, 2, 1, 3),
            values.permute(0, 2, 1, 3),
            attn_mask
        )[0]

        return res.permute(0, 2, 1, 3).contiguous(), None


# ======================================================
# FullAttention
# ======================================================
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # >>> MOD
    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None, time_bias=None):

        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


# ======================================================
# ProbAttention
# ======================================================
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # >>> MOD
    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None, time_bias=None):

        B, L_Q, H, D = queries.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scale = self.scale or 1. / sqrt(D)
        scores = scores * scale

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, values)

        return context.transpose(2, 1).contiguous(), None


# ======================================================
# AttentionLayer（🔥 关键修改点）
# ======================================================
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    # >>> MOD: 增加 time_bias，并向 inner_attention 透传
    def forward(self, queries, keys, values, attn_mask,
                tau=None, delta=None, time_bias=None):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            time_bias=time_bias    # 🔥 生死点
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ======================================================
# ReformerLayer（不动）
# ======================================================
class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None,
                 causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
        return torch.cat(
            [queries, torch.zeros(B, fill_len, C, device=queries.device)],
            dim=1
        )

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
