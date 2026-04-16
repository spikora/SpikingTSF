"""
Spikingformer for long-term time series forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Copyright (c) Microsoft Corporation.
  Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks",
  ICML 2024.  https://github.com/microsoft/SeqSNN  (SeqSNN/network/snn/spikingformer.py)

Adaptation notes:
  - Removed registry/runner dependency; wraps backbone into a standalone Model(configs)
  - Added per-sample instance normalisation (TSLib protocol)
  - Output projection included inside the model: (B, pred_len, D) directly
  - Uses spikingjelly activation_based API with functional.set_step_mode

Key difference from Spikformer: adds ConvPE (convolutional positional encoding),
a 1-D conv over the sequence axis followed by a LIF gate — this injects local
temporal context into the token representations before the SSA blocks.

Architecture:
    Input (B, L, D)
      ↓  per-sample instance norm
    ConvEncoder → (T, B, D, L) → transpose → (T, B, L, D)
      ↓  ConvPE  (residual conv-based positional encoding)
    Linear(D → d_model) + BN → (T, B, L, d_model)
      ↓  N × [SSA + MLP] Block
    Mean over T → (B, L, d_model)
    Mean over L → (B, d_model)
    Linear(d_model → pred_len × D) → (B, pred_len, D)
      ↓  denorm
    Output (B, pred_len, D)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from models.layers.spike_attention import Block


class ConvPE(nn.Module):
    """Convolutional positional encoding — residual Conv1d + BN + LIF."""

    def __init__(self, d_model: int, tau: float, dropout: float = 0.1):
        super().__init__()
        self.rpe_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(tau=tau, detach_reset=True,
                                      surrogate_function=surrogate.ATan())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D)
        T, B, L, D = x.shape
        x_flat = x.flatten(0, 1)               # (TB, L, D)
        feat = x_flat.transpose(1, 2)          # (TB, D, L)
        feat = self.rpe_conv(feat)             # (TB, D, L)
        feat = self.rpe_bn(feat).reshape(T, B, D, L).contiguous()
        feat = self.rpe_lif(feat)              # (T, B, D, L)
        feat = feat.flatten(0, 1)              # (TB, D, L)
        feat = self.dropout(feat)
        feat = feat.permute(0, 2, 1)          # (TB, L, D)
        x_flat = x_flat + feat
        return x_flat.reshape(T, B, L, D)


class Model(nn.Module):
    """
    Spikingformer: Spikformer with convolutional positional encoding.

    Args (from configs namespace):
        seq_len      : input length L
        pred_len     : forecast horizon H
        enc_in       : number of input variables D
        T            : SNN time steps (used as encoder output channels)
        tau          : LIF membrane time constant
        levels       : number of transformer blocks
        d_model      : attention / embedding dimension
        n_heads      : number of attention heads
        d_ff         : feedforward hidden dim (default 4 × d_model)
        common_thr   : LIF spike threshold (default 1.0)
        qk_scale     : attention score scale (default 0.125)
        dropout      : dropout (default 0.1)
    """

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        D = configs.enc_in
        T = configs.T
        tau = configs.tau
        depths = configs.levels
        d_model = getattr(configs, 'd_model', 256)
        d_ff = getattr(configs, 'd_ff', d_model * 4)
        n_heads = getattr(configs, 'n_heads', 8)
        common_thr = getattr(configs, 'common_thr', 1.0)
        qk_scale = getattr(configs, 'qk_scale', 0.125)
        dropout = getattr(configs, 'dropout', 0.1)

        self.pred_len = pred_len
        self.D = D
        self.T = T

        # Temporal encoder: conv2d(1, T) treated as spike-rate encoder
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(1, T, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(T),
        )
        self.enc_lif = neuron.LIFNode(tau=tau, detach_reset=True,
                                      surrogate_function=surrogate.ATan())

        self.conv_pe = ConvPE(d_model=D, tau=tau, dropout=dropout)

        self.encoder = nn.Linear(D, d_model)
        self.init_bn = nn.BatchNorm1d(d_model)

        self.blocks = nn.ModuleList([
            Block(length=seq_len, tau=tau, common_thr=common_thr,
                  dim=d_model, d_ff=d_ff, heads=n_heads, qk_scale=qk_scale)
            for _ in range(depths)
        ])

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, pred_len * D)

        self._init_weights()
        # Set all neurons to multi-step mode
        functional.set_step_mode(self, step_mode='m')

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        functional.reset_net(self)

        B, L, D = x.shape

        # Temporal encoding
        enc_in = x.permute(0, 2, 1).unsqueeze(1)         # (B, 1, D, L)
        enc = self.temporal_encoder(enc_in)               # (B, T, D, L)
        enc = enc.permute(1, 0, 2, 3)                     # (T, B, D, L)
        x = self.enc_lif(enc)                             # (T, B, D, L)
        x = x.transpose(-2, -1)                           # (T, B, L, D)

        T = x.shape[0]

        # Convolutional positional encoding
        x = self.conv_pe(x)                               # (T, B, L, D)

        # Project D → d_model with BN
        x_flat = x.flatten(0, 1)                          # (TB, L, D)
        x_flat = self.encoder(x_flat)                     # (TB, L, d_model)
        x_flat = self.init_bn(x_flat.transpose(-1, -2)).transpose(-1, -2)
        x = x_flat.reshape(T, B, L, -1).contiguous()

        for blk in self.blocks:
            x = blk(x)                                    # (T, B, L, d_model)

        x = x.mean(0)                                     # (B, L, d_model)
        x = self.dropout(x)
        x = x.mean(1)                                     # (B, d_model)

        x = self.out_proj(x)                              # (B, pred_len * D)
        x = x.reshape(B, self.pred_len, self.D)

        return x * std + mean
