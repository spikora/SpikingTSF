"""
Spiking spike encoders — adapted from SeqSNN (Microsoft, MIT License).

Copyright (c) Microsoft Corporation.
https://github.com/microsoft/SeqSNN  (SeqSNN/module/encoding/spikingjelly/encoder.py)

Three encoder variants:
  ConvEncoder  — 2-D conv over (C, L) then LIF; produces (T, B, C, L)
  DeltaEncoder — temporal delta (first-order difference) then LIF; produces (T, B, C, L)
  RepeatEncoder — repeat input T times then LIF; produces (T, B, C, L)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron


class ConvEncoder(nn.Module):
    """Conv2D encoder: (B, L, C) -> (T, B, C, L)."""

    def __init__(self, output_size: int, tau: float = 2.0, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, output_size,
                      kernel_size=(1, kernel_size), stride=1,
                      padding=(0, kernel_size // 2)),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = x.permute(0, 2, 1).unsqueeze(1)   # (B, 1, C, L)
        x = self.encoder(x)                    # (B, T, C, L)
        x = x.permute(1, 0, 2, 3)             # (T, B, C, L)
        return self.lif(x)                     # (T, B, C, L)


class DeltaEncoder(nn.Module):
    """Temporal delta encoder: (B, L, C) -> (T, B, C, L)."""

    def __init__(self, output_size: int, tau: float = 2.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:, :] - x[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, C, L)
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)               # (B, C, L, 1)
        enc = self.enc(delta)                            # (B, C, L, T)
        enc = enc.permute(3, 0, 1, 2)                   # (T, B, C, L)
        return self.lif(enc)                             # (T, B, C, L)


class RepeatEncoder(nn.Module):
    """Repeat encoder: tile input T times then LIF; (B, L, C) -> (T, B, C, L)."""

    def __init__(self, output_size: int, tau: float = 2.0):
        super().__init__()
        self.out_size = output_size
        self.lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        ones = [1] * len(x.shape)
        x = x.repeat([self.out_size] + ones)  # (T, B, L, C)
        x = x.permute(0, 1, 3, 2)            # (T, B, C, L)
        return self.lif(x)                    # (T, B, C, L)
