import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.TSSA import AttentionTSSA
from utils.tools import standard_scaler
from utils.timefeatures import time_features_from_frequency_str
import numpy as np


class AEFINDecomposer(nn.Module):
    """
    Adaptive Explicit Frequency Interaction Network (改进版)
    作为可学习时序分解器，放在standard_scaler之后，embedding之前

    修复：
    1. 添加residual连接 - 保留原始低频趋势和恒等路径
    2. Variable-level gate - 不同变量使用不同频率权重
    """

    def __init__(self, seq_len, d_model, n_freq=3):
        super().__init__()
        self.n_freq = n_freq

        # 可学习频域滤波器（Conv1D = learnable frequency response）
        # 使用不同kernel_size的卷积来捕获不同频率成分
        self.filters = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=k,
                padding=k//2
            )
            for k in [3, 7, 15][:n_freq]  # 低中高频滤波器
        ])

        # 🔧 修复1：频段权重改为variable-level（每个变量独立）
        # 输入: [B, N, F], 输出: [B, N, F]
        self.freq_gate = nn.Sequential(
            nn.Linear(n_freq, n_freq),
            nn.Softmax(dim=-1)
        )

        # 🔧 修复2：添加residual门控参数
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值，让模型逐渐学习

    def forward(self, x):
        """
        x: [B, L, N] - 标准化后的输入
        return: [B, L, N] - 经过频域分解和重构的输出（带residual）
        """
        B, L, N = x.shape
        x_ = x.permute(0, 2, 1)        # [B, N, L] - 为了使用Conv1D

        freq_components = []
        for conv in self.filters:
            # 对每个变量通道单独进行卷积滤波
            comp = conv(x_.reshape(B*N, 1, L))  # [B*N, 1, L]
            comp = comp.reshape(B, N, L)        # [B, N, L]
            freq_components.append(comp)

        # 堆叠所有频段成分: [B, N, L, F]
        freq_stack = torch.stack(freq_components, dim=-1)

        # 🔧 修复2：计算variable-level频段能量 [B, N, F]
        energy = freq_stack.mean(dim=2)         # [B, N, F] - 每个变量的频段能量
        weights = self.freq_gate(energy)         # [B, N, F] - 每个变量的频率权重

        # 🔧 修复1：加权融合 + residual连接
        fused_freq = (freq_stack * weights.unsqueeze(2)).sum(dim=-1)  # [B, N, L]

        # 自适应residual权重 (0~1之间)
        alpha = torch.sigmoid(self.alpha)

        # 最终输出 = 原始信号 + 门控的频域增强
        out = x_ + alpha * fused_freq  # [B, N, L]

        return out.permute(0, 2, 1)  # [B, L, N] - 转回原始维度


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    # =====================================================
    # 🔧 MOD-1: Time2Vec（支持任意 input_dim）
    # =====================================================
    class Time2Vec(nn.Module):
        def __init__(self, dim, input_dim=1):
            super().__init__()
            self.w0 = nn.Linear(input_dim, 1)
            self.w = nn.Linear(input_dim, dim - 1)
            self.v = nn.Linear(input_dim, dim - 1)

        def forward(self, t):
            # t: [B, L, input_dim]
            v1 = self.w0(t)                          # [B, L, 1]
            v2 = torch.sin(self.w(t) + self.v(t))    # [B, L, dim-1]
            return torch.cat([v1, v2], dim=-1)       # [B, L, dim]
    # =====================================================

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model

        # =====================================================
        # 🔧 MOD-2: Time2Vec 初始化（timeF, 根据频率动态确定维度）
        # =====================================================
        self.use_time2vec = getattr(configs, 'use_time2vec', True)
        if self.use_time2vec:
            # 动态计算时间特征维度 (例如: '15min' -> 5维, 'h' -> 4维)
            time_feature_classes = time_features_from_frequency_str(configs.freq)
            input_dim = len(time_feature_classes)
            self.time_embed = self.Time2Vec(
                dim=configs.d_model,
                input_dim=input_dim
            )
        # =====================================================

        # =====================================================
        # 🔧 MOD-A: AEFIN Decomposer 初始化
        # =====================================================
        self.use_aefin = getattr(configs, 'use_aefin', True)
        if self.use_aefin:
            self.aefin = AEFINDecomposer(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                n_freq=3  # 使用3个频段：低中高频
            )
        # =====================================================

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # =====================================================
        # Encoder with TSSA
        # =====================================================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AttentionTSSA(
                            dim=configs.d_model,
                            num_heads=configs.n_heads,
                            attn_drop=configs.dropout,
                            proj_drop=configs.dropout
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # =====================================================
        # 🔧 MOD-5: 输出对齐层 (类似Minusformer)
        # =====================================================
        self.d_block = getattr(configs, 'd_block', configs.d_model)
        if self.d_block != configs.pred_len:
            self.align = nn.Linear(self.d_block, configs.pred_len)
        else:
            self.align = nn.Identity()
        self.projector = nn.Linear(configs.d_model, self.d_block, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # =====================================================
        # 🔧 MOD-6: 改进的标准化 (类似Minusformer)
        # =====================================================
        if self.use_norm:
            # 使用standard_scaler工具，类似Minusformer
            x_enc_permuted = x_enc.permute(0, 2, 1)  # [B, L, N] -> [B, N, L]
            scaler = standard_scaler(x_enc_permuted)
            x_enc_norm = scaler.transform(x_enc_permuted)
            x_enc_norm = x_enc_norm.permute(0, 2, 1)  # [B, N, L] -> [B, L, N]
        else:
            x_enc_norm = x_enc
            scaler = None

        B, L, N = x_enc.shape

        # =====================================================
        # 🔧 MOD-3: Time2Vec → attention time_bias（核心）
        # =====================================================
        time_bias = None
        if self.use_time2vec and x_mark_enc is not None:
            # x_mark_enc: [B, L, 4]
            time_emb = self.time_embed(x_mark_enc.float())   # [B, L, d_model]

            # 预测锚点时间（最后一个 encoder 时间）
            t_anchor = time_emb[:, -1, :]                    # [B, d_model]

            # 最稳定的第一版：scalar phase bias
            time_bias = t_anchor.mean(dim=-1, keepdim=True)  # [B, 1]
            time_bias = time_bias.view(B, 1, 1)              # [B, 1, 1]
        # =====================================================

        # =====================================================
        # 🔥 MOD-B: AEFIN 显式非平稳建模（时序分解器）
        # =====================================================
        if self.use_aefin:
            x_enc_norm = self.aefin(x_enc_norm)  # [B, L, N] -> [B, L, N]
        # =====================================================

        # =====================================================
        # 🔧 MOD-7: 改进的嵌入处理 (结合Time2Vec创新)
        # =====================================================
        enc_out = self.enc_embedding(x_enc_norm, x_mark_enc)  # [B, N, d_model]

        # =====================================================
        # 🔧 MOD-4: 把 time_bias 送入 Encoder（而不是加到 enc_out）
        # =====================================================
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None,
            time_bias=time_bias
        )
        # =====================================================

        # =====================================================
        # 🔧 MOD-8: 改进的输出处理 (类似Minusformer)
        # =====================================================
        # 先投影到d_block维度
        dec_out = self.projector(enc_out)                     # [B, N, d_block]
        # 然后对齐到pred_len
        dec_out = self.align(dec_out)                         # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]          # [B, pred_len, N]

        # =====================================================
        # 🔧 MOD-9: 改进的去标准化
        # =====================================================
        if self.use_norm and scaler is not None:
            # 使用scaler的逆变换，类似Minusformer
            dec_out_permuted = dec_out.permute(0, 2, 1)      # [B, pred_len, N] -> [B, N, pred_len]
            dec_out = scaler.inverted(dec_out_permuted)      # [B, N, pred_len]
            dec_out = dec_out.permute(0, 2, 1)              # [B, pred_len, N]

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
