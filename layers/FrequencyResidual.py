import torch
import torch.nn as nn
import torch.nn.functional as F


def dct(x: torch.Tensor) -> torch.Tensor:
    """
    A simple DCT-like transform using FFT trick.
    Input:
        x: [B, L]
    Output:
        Tensor [B, L] (real part)
    """
    N = x.shape[-1]
    # pack even then reversed odds
    v = torch.cat([x[:, ::2], x[:, 1::2].flip(dims=[1])], dim=1)
    V = torch.fft.fft(v)
    return V.real[:, :N]


class FrequencyResidual(nn.Module):
    """
    Frequency Residual branch that extracts low-K frequency coefficients
    per channel and projects them to the model dimension.

    Inputs:
        seq_len: original time length L
        d_model: target embedding dim
        channels: number of input variates (N)
        k: number of frequency coefficients to keep (low-freq)
        method: 'dct' or 'fft'
    Forward:
        x: [B, L, N] -> returns [B, N, d_model] to match inverted embedding
    """

    def __init__(self, seq_len: int, d_model: int, channels: int, k: int = 8, method: str = "dct"):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # channels is kept for compatibility but not strictly enforced at runtime
        self.channels = channels
        self.k = min(k, seq_len)
        self.method = method

        # project frequency vector of length k -> d_model (applied per channel)
        self.proj = nn.Linear(self.k, d_model)
        # global learnable scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N]
        returns: [B, N, d_model]
        """
        B, L, N = x.shape
        # Accept runtime channel count (may include concatenated time features).
        # Keep self.channels for backward compatibility but do not enforce equality here.

        # reshape to compute transform per (batch * channel)
        x_t = x.permute(0, 2, 1).contiguous()      # [B, N, L]
        x_resh = x_t.view(-1, L)                  # [B*N, L]

        if self.method.lower() == "dct":
            freq = dct(x_resh)                    # [B*N, L]
        else:
            # fft: take real part
            V = torch.fft.fft(x_resh)
            freq = V.real[:, :L]

        # take low-k coefficients (first k)
        freq_k = freq[:, : self.k]                # [B*N, k]
        freq_k = freq_k.view(B, N, self.k)        # [B, N, k]

        # project k -> d_model per channel
        freq_proj = self.proj(freq_k)             # [B, N, d_model]

        return self.alpha * freq_proj


