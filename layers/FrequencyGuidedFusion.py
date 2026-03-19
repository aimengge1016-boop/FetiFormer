import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """
    多尺度特征构建：
    - 原尺度
    - 平滑尺度 (AvgPool)
    - 局部尺度 (Depthwise Conv)
    """
    def __init__(self, d_model):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

    def forward(self, x):
        # x: [B, N, D]
        x_t = x.transpose(1, 2)  # [B, D, N]
        x_smooth = self.avg_pool(x_t)
        x_local = self.local_conv(x_t)
        return x, x_smooth.transpose(1, 2), x_local.transpose(1, 2)


class FrequencyGuidedFusion(nn.Module):
    """
    频域引导的多尺度特征融合
    输入:
        enc_out: [B, N, D] 经过 embedding 的特征
        x_enc:   [B, L, N] 原始时间序列
    输出:
        [B, N, D] 融合特征
    """
    def __init__(self, d_model):
        super().__init__()
        self.ms_block = MultiScaleBlock(d_model)

        self.freq_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, enc_out, x_enc):
        B_enc, N_enc, D = enc_out.shape
        B_x, L, N_vars = x_enc.shape

        # ===== 频域统计 =====
        # x_enc: [B, L, N_vars]
        fft = torch.fft.rfft(x_enc, dim=1)
        freq_energy = torch.abs(fft).mean(dim=1)     # [B, N_vars]
        freq_energy_in = freq_energy.unsqueeze(-1)   # [B, N_vars, 1]

        weights = self.freq_gate(freq_energy_in)     # [B, N_vars, 3]

        # ===== 多尺度特征 =====
        # If enc_out has extra tokens (e.g., time tokens), only fuse the first N_vars
        if N_enc >= N_vars:
            main = enc_out[:, :N_vars, :]                    # [B, N_vars, D]
            tail = enc_out[:, N_vars:, :] if N_enc > N_vars else None

            x0, x1, x2 = self.ms_block(main)                # each: [B, N_vars, D]

            w0 = weights[..., 0:1]                          # [B, N_vars, 1]
            w1 = weights[..., 1:2]
            w2 = weights[..., 2:3]

            fused_main = w0 * x0 + w1 * x1 + w2 * x2      # [B, N_vars, D]

            if tail is not None:
                out = torch.cat([fused_main, tail], dim=1)  # [B, N_enc, D]
            else:
                out = fused_main
        else:
            # enc_out has fewer tokens than variables in x_enc — trim weights
            main = enc_out
            x0, x1, x2 = self.ms_block(main)             # each: [B, N_enc, D]
            weights_trim = weights[:, :N_enc, :]
            w0 = weights_trim[..., 0:1]
            w1 = weights_trim[..., 1:2]
            w2 = weights_trim[..., 2:3]
            out = w0 * x0 + w1 * x1 + w2 * x2

        return out


