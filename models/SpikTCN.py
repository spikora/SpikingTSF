"""
Spiking Temporal Convolutional Network for long-term time series forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks", ICML 2024.
  https://github.com/microsoft/SeqSNN

Architecture changes vs. SeqSNN SpikeTemporalConvNet2D:
  - Standalone (no runner/encoder registry dependency)
  - Channel-independent processing (one TCN per variate, like PatchTST / DLinear philosophy)
  - Uses spikingjelly clock_driven API (consistent with SpikF / iSpikformer in this codebase)
  - Returns (B, pred_len, D) directly without a separate prediction head wrapper
"""

import torch
from torch import nn
from torch.nn.utils import weight_norm
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SpikeTCNBlock(nn.Module):
    """One residual dilated-causal-conv block with two spiking LIF layers."""

    def __init__(self, channels, kernel_size, dilation, T, tau, dropout=0.1):
        super().__init__()
        self.T = T
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')

        self.conv2 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')

        self.dropout = nn.Dropout(dropout)

    def _spike(self, lif, x):
        # x: (BD, C, L) → expand to T steps → (T, BD, C, L) → LIF → mean over T
        t = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        return lif(t).mean(0)

    def forward(self, x):
        # x: (BD, C, L)
        out = self.chomp1(self.bn1(self.conv1(x)))
        out = self._spike(self.lif1, out)
        out = self.dropout(out)
        out = self.chomp2(self.bn2(self.conv2(out)))
        out = self._spike(self.lif2, out)
        out = self.dropout(out)
        return out + x  # residual connection


class Model(nn.Module):
    """
    SpikTCN: channel-independent spiking TCN.

    Each of the D input variables is processed as an independent 1-D sequence.
    Receptive field doubles per level via dilation = 2^i.

    Args (from configs namespace):
        seq_len    : input length L
        pred_len   : forecast horizon H
        enc_in     : number of input variables D
        patch_dim  : TCN hidden channels (d_model)
        T          : SNN time-steps (static input repeated T times)
        levels     : number of TCN blocks
        tau        : LIF membrane time constant (τ > 1)
        kernel_size: TCN kernel size (default 3)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        D = configs.enc_in
        T = configs.T
        tau = configs.tau
        patch_dim = configs.patch_dim
        levels = configs.levels
        kernel_size = getattr(configs, 'kernel_size', 3)
        dropout = getattr(configs, 'dropout', 0.1)

        # 1-channel embedding: (BD, 1, L) → (BD, patch_dim, L)
        self.input_proj = nn.Conv1d(1, patch_dim, kernel_size=1)

        # Stacked TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            SpikeTCNBlock(patch_dim, kernel_size, 2 ** i, T, tau, dropout)
            for i in range(levels)
        ])

        self.bn = nn.BatchNorm1d(D)
        # Project last-position hidden state to pred_len
        self.out_proj = nn.Linear(patch_dim, configs.pred_len)

    def forward(self, x):
        # x: (B, L, D)
        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        B, L, D = x.shape

        # Channel-independent: (B, L, D) → (BD, 1, L)
        x = x.permute(0, 2, 1).reshape(B * D, 1, L)
        x = self.input_proj(x)  # (BD, patch_dim, L)

        for block in self.blocks:
            x = block(x)

        # Causal TCN: output at last position captures all history
        x = x[:, :, -1]  # (BD, patch_dim)
        x = self.out_proj(x)  # (BD, pred_len)

        x = x.reshape(B, D, self.pred_len).permute(0, 2, 1)  # (B, pred_len, D)

        return x * std + mean
