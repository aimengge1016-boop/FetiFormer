import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.TSSA import AttentionTSSA
import numpy as np


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

        # =====================================================
        # 🔧 MOD-2: Time2Vec 初始化（timeF, freq='h' → 4维）
        # =====================================================
        self.use_time2vec = getattr(configs, 'use_time2vec', True)
        if self.use_time2vec:
            self.time_embed = self.Time2Vec(
                dim=configs.d_model,
                input_dim=4   # hour, weekday, day, month
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

        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.pred_len) 
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # =====================================================
        # Normalization
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
        # Inverted Embedding
        # =====================================================
        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # [B, N, d_model]

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
