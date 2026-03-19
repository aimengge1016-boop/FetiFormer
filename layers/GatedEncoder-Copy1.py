import torch
import torch.nn as nn
from .ProgressiveGate import ProgressiveGate


class GatedEncoder(nn.Module):
    """
    Wraps a list of EncoderLayer-like modules and applies a ProgressiveGate
    after each layer. Does not modify the EncoderLayer internals.
    """
    def __init__(self, encoder_layers, d_model):
        super().__init__()
        # encoder_layers: a list of EncoderLayer instances (already constructed)
        self.layers = nn.ModuleList(encoder_layers)
        self.gates = nn.ModuleList(
            [ProgressiveGate(d_model) for _ in range(len(self.layers))]
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None, time_bias=None):
        """
        x: [B, N, D]
        returns: x, gates_list
        Note: time_bias is accepted for API compatibility but not forwarded
        because EncoderLayer.forward signature may not accept it.
        """
        gates = []
        for layer, gate in zip(self.layers, self.gates):
            residual = x
            # call the original layer (preserve its interface)
            try:
                x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            except TypeError:
                # fallback: call without named args if some layers expect positional
                x, attn = layer(x)

            x, g = gate(x, residual)
            gates.append(g)

        return x, gates


