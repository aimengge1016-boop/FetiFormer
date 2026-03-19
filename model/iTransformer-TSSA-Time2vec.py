import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.TSSA import AttentionTSSA
from utils.timefeatures import time_features_from_frequency_str
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    # =====================================================
    # 🔧 MOD-1: Time2Vec（支持任意 input_dim）
    # =====================================================
    class Time2Vec(nn.Module):
        def __init__(self, dim, input_dim=1, freq='h'):
            super().__init__()
            self.freq = freq
            self.w0 = nn.Linear(input_dim, 1)
            self.w = nn.Linear(input_dim, dim - 1)
            self.v = nn.Linear(input_dim, dim - 1)

            # 🔧 增强版：为高频数据添加周期性编码
            if freq == '15min':
                # 15分钟数据的特殊周期：96点/天，4点/小时
                self.cycle_embed = nn.Linear(input_dim, dim // 2)
                self.daily_pe = self._get_periodic_encoding(96)  # 96个15分钟点构成一天
                self.hourly_pe = self._get_periodic_encoding(4)  # 4个15分钟点构成一小时

        def _get_periodic_encoding(self, period):
            """生成周期性位置编码"""
            position = torch.arange(period).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, 64, 2) * (-torch.log(torch.tensor(10000.0)) / 64))
            pe = torch.zeros(period, 64)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        def forward(self, t):
            # t: [B, L, input_dim]
            v1 = self.w0(t)                          # [B, L, 1]
            v2 = torch.sin(self.w(t) + self.v(t))    # [B, L, dim-1]

            # 🔧 增强版：为15分钟数据添加周期性特征
            if hasattr(self, 'cycle_embed') and self.freq == '15min':
                # 添加每日和小时周期特征
                B, L, _ = t.shape
                daily_pos = torch.arange(L) % 96  # 96个15分钟点 = 1天
                hourly_pos = torch.arange(L) % 4   # 4个15分钟点 = 1小时

                daily_pe = self.daily_pe[daily_pos].to(t.device).unsqueeze(0).expand(B, -1, -1)
                hourly_pe = self.hourly_pe[hourly_pos].to(t.device).unsqueeze(0).expand(B, -1, -1)

                cycle_features = self.cycle_embed(t)  # [B, L, dim//2]
                enhanced_v2 = torch.cat([v2, cycle_features, daily_pe, hourly_pe], dim=-1)
                return torch.cat([v1, enhanced_v2], dim=-1)
            else:
                return torch.cat([v1, v2], dim=-1)       # [B, L, dim]
    # =====================================================

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # =====================================================
        # 🔧 MOD-2: Time2Vec 初始化（timeF, 根据频率动态确定维度）
        # =====================================================
        self.use_time2vec = getattr(configs, 'use_time2vec', True)
        if self.use_time2vec:
            # 动态计算时间特征维度 (15min -> 5维, h -> 4维, d -> 3维)
            time_feature_classes = time_features_from_frequency_str(configs.freq)
            input_dim = len(time_feature_classes)
            self.time_embed = self.Time2Vec(
                dim=configs.d_model,
                input_dim=input_dim,
                freq=configs.freq
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
        # 🔧 MOD-IRON: 特征级注意力层（针对IRON数据集的7个特征）
        # =====================================================
        self.feature_attention = None
        if getattr(configs, 'use_feature_attention', True):
            self.feature_attention = nn.MultiheadAttention(
                embed_dim=configs.d_model,
                num_heads=configs.n_heads // 2,  # 使用较少的头来聚焦特征关系
                dropout=configs.dropout,
                batch_first=True
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

        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len) 
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape
        # =====================================================
        # 🔧 MOD-IRON: 自适应归一化（针对非平稳IRON数据集）
        # =====================================================
        if self.use_norm:
            # 基础归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

            # 🔧 增强版：针对IRON数据集的负载类型自适应
            # 检测可能的负载类型变化（通过特征模式识别）
            if N >= 7:  # IRON数据集有7个特征
                # 计算特征间的相关性变化作为负载类型指标
                feature_corr = torch.corrcoef(x_enc.permute(2, 0, 1).reshape(N, -1))
                load_type_indicator = torch.std(feature_corr, dim=0).mean()

                # 如果负载类型变化明显，使用更强的归一化
                if load_type_indicator > 0.3:  # 经验阈值
                    # 对每个特征分别进行更强的局部归一化
                    for i in range(N):
                        feature_data = x_enc[:, :, i:i+1]  # [B, L, 1]
                        local_means = feature_data.mean(1, keepdim=True)
                        local_stdev = torch.sqrt(torch.var(feature_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
                        x_enc[:, :, i:i+1] = (feature_data - local_means) / (local_stdev + 1e-5)

        B, L, _ = x_enc.shape

        # =====================================================
        # 🔧 MOD-3: Time2Vec → attention time_bias（核心）
        # =====================================================
        time_bias = None
        if self.use_time2vec and x_mark_enc is not None:
            # x_mark_enc: [B, L, input_dim] (根据频率动态确定)
            time_emb = self.time_embed(x_mark_enc.float())   # [B, L, d_model]

            # 预测锚点时间（最后一个 encoder 时间）
            t_anchor = time_emb[:, -1, :]                    # [B, d_model]

            # 最稳定的第一版：scalar phase bias
            time_bias = t_anchor.mean(dim=-1, keepdim=True)  # [B, 1]
            time_bias = time_bias.view(B, 1, 1)              # [B, 1, 1]
        # =====================================================

        # =====================================================
        # Inverted Embedding
        # =====================================================
        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # [B, N, d_model]

        # =====================================================
        # 🔧 MOD-IRON: 特征级注意力（针对IRON数据集的多变量关系）
        # =====================================================
        # 为IRON数据集（7个特征）添加特征间的注意力机制
        if N == 7 and hasattr(self, 'feature_attention'):
            # 特征级自注意力：捕捉7个特征间的复杂关系
            feature_attn_out, _ = self.feature_attention(enc_out)  # [B, N, d_model]
            enc_out = enc_out + feature_attn_out  # 残差连接

        # =====================================================
        # 🔧 MOD-4: 把 time_bias 送入 Encoder（而不是加到 enc_out）
        # =====================================================
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None,
            time_bias=time_bias
        )
        # =====================================================

        # Projection
        dec_out = self.projector(enc_out)                     # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]          # [B, pred_len, N]

        # =====================================================
        # De-normalization
        # =====================================================
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) \
                      + means[:, 0, :].unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
