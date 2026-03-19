import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    # ✅ ========== 修正：Time2Vec 支持任意 input_dim ==========
    class Time2Vec(nn.Module):
        def __init__(self, dim, input_dim=1):
            super().__init__()
            self.dim = dim
            self.input_dim = input_dim
            # Linear trend term: [input_dim] -> [1]
            self.w0 = nn.Linear(input_dim, 1)
            # Periodic terms: [input_dim] -> [dim - 1]
            self.w = nn.Linear(input_dim, dim - 1)
            self.v = nn.Linear(input_dim, dim - 1)

        def forward(self, t):
            # t: [B, L, input_dim]
            v1 = self.w0(t)  # [B, L, 1]
            v2 = torch.sin(self.w(t) + self.v(t))  # [B, L, dim - 1]
            return torch.cat([v1, v2], dim=-1)  # [B, L, dim]
    # ✅ =========================================

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # ✅ ========== 关键修正：显式指定 input_dim=4 ==========
        self.use_time2vec = getattr(configs, 'use_time2vec', True)
        if self.use_time2vec:
            # 因为使用 --embed timeF --freq 'h'，x_mark_enc 维度为 4
            # (hour, day-of-week, day-of-month, month)
            self.time_embed = self.Time2Vec(configs.d_model, input_dim=4)
        # ✅ =========================================
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy
        
        # -------- Encoder（使用标准 FullAttention） --------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads,
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

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len) 
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape  # B: batch_size; L: seq_len; N: variate count

        # ✅ ========== 正确处理 x_mark_enc (shape [B, L, 4]) ==========
        time_emb = None
        if self.use_time2vec and x_mark_enc is not None:
            # x_mark_enc is [B, L, 4] from time_features(freq='h')
            timestamps = x_mark_enc.float()  # Ensure float type
            time_emb = self.time_embed(timestamps)  # [B, L, d_model]
        # ✅ =========================================

        # Inverted embedding: [B, L, N] -> [B, N, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, d_model]

        # ✅ Fuse time embedding (broadcast over variates)
        if self.use_time2vec and time_emb is not None:
            time_summary = time_emb.mean(dim=1, keepdim=True)  # [B, 1, d_model]
            enc_out = enc_out + time_summary  # [B, N, d_model] + [B, 1, d_model]

        # Encoder processing
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Project to prediction length
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        if self.use_norm:
            # De-normalization
            stdev_expanded = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            means_expanded = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * stdev_expanded + means_expanded

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]