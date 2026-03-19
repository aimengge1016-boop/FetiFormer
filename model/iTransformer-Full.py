import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.TSSA import AttentionTSSA

# >>> NEW: 频域引导多尺度 + 渐进式门控
from layers.FrequencyGuidedFusion import FrequencyGuidedFusion
from layers.GatedEncoder import GatedEncoder


class Model(nn.Module):
    """
    iTransformer + Time2Vec + TSSA
    + Frequency-Guided Multi-Scale Fusion
    + Progressive Gated Encoder
    """

    # =====================================================
    # MOD-1: Time2Vec（保持你原实现，未改）
    # =====================================================
    class Time2Vec(nn.Module):
        def __init__(self, dim, input_dim=1):
            super().__init__()
            self.w0 = nn.Linear(input_dim, 1)
            self.w = nn.Linear(input_dim, dim - 1)
            self.v = nn.Linear(input_dim, dim - 1)

        def forward(self, t):
            v1 = self.w0(t)
            v2 = torch.sin(self.w(t) + self.v(t))
            return torch.cat([v1, v2], dim=-1)
    # =====================================================

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # =====================================================
        # MOD-2: Time2Vec 初始化（保持原逻辑）
        # =====================================================
        self.use_time2vec = getattr(configs, 'use_time2vec', True)
        if self.use_time2vec:
            self.time_embed = self.Time2Vec(
                dim=configs.d_model,
                input_dim=4
            )

        # =====================================================
        # Embedding（未改）
        # =====================================================
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # =====================================================
        # >>> NEW-1: 频域引导多尺度融合模块
        # =====================================================
        self.freq_fusion = FrequencyGuidedFusion(
            d_model=configs.d_model
        )

        # =====================================================
        # >>> MOD-3: 使用 GatedEncoder 替换原 Encoder
        # =====================================================
        encoder_layers = [
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
        ]

        self.encoder = GatedEncoder(
            encoder_layers=encoder_layers,
            d_model=configs.d_model
        )

        # =====================================================
        # Projection（未改）
        # =====================================================
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # =====================================================
        # Normalization（未改）
        # =====================================================
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape

        # =====================================================
        # MOD-4: Time2Vec → attention time_bias（未改）
        # =====================================================
        time_bias = None
        if self.use_time2vec and x_mark_enc is not None:
            time_emb = self.time_embed(x_mark_enc.float())   # [B,L,D]
            t_anchor = time_emb[:, -1, :]
            time_bias = t_anchor.mean(dim=-1, keepdim=True)
            time_bias = time_bias.view(B, 1, 1)

        # =====================================================
        # Inverted Embedding
        # =====================================================
        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # [B,N,D]

        # =====================================================
        # >>> NEW-2: Frequency-Guided Multi-Scale Fusion
        # =====================================================
        enc_out = self.freq_fusion(enc_out, x_enc)

        # =====================================================
        # >>> MOD-5: Gated Encoder Forward
        # =====================================================
        enc_out, gate_values = self.encoder(
            enc_out,
            attn_mask=None,
            time_bias=time_bias
        )

        # =====================================================
        # Projection
        # =====================================================
        dec_out = self.projector(enc_out)                     # [B,N,pred_len]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]          # [B,pred_len,N]

        # =====================================================
        # De-normalization
        # =====================================================
        if self.use_norm:
            dec_out = (
                dec_out * stdev[:, 0, :].unsqueeze(1)
                + means[:, 0, :].unsqueeze(1)
            )

        return dec_out, gate_values

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, aux = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], aux
        else:
            return dec_out[:, -self.pred_len:, :]
