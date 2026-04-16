"""
Spikformer for long-term time series forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Copyright (c) Microsoft Corporation.
  Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks",
  ICML 2024.  https://github.com/microsoft/SeqSNN  (SeqSNN/network/snn/spikformer.py)

Adaptation notes:
  - Removed registry/runner dependency; wraps backbone into a standalone Model(configs)
  - Added per-sample instance normalisation (TSLib protocol) instead of SeqSNN's global norm
  - Output projection included inside the model: (B, pred_len, D) directly
  - Uses spikingjelly activation_based API (step_mode='m') throughout

Architecture:
    Input (B, L, D)
      ↓  per-sample instance norm
    ConvEncoder → (T, B, D, L) → transpose → (T, B, L, D)
      ↓  Linear(D → d_model) + BN + LIF
      ↓  N × [SSA + MLP] Block  (tokens = time steps L)
    Mean over T → (B, L, d_model)
    Mean over L → (B, d_model)        [global sequence embedding]
    Linear(d_model → pred_len × D) → reshape (B, pred_len, D)
      ↓  denorm
    Output (B, pred_len, D)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from models.layers.spike_attention import Block
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder, RepeatEncoder

_ENCODERS = {'conv': ConvEncoder, 'delta': DeltaEncoder, 'repeat': RepeatEncoder}


class Model(nn.Module):
    """
    Spikformer: temporal-token spiking transformer.

    Args (from configs namespace):
        seq_len      : input length L
        pred_len     : forecast horizon H
        enc_in       : number of input variables D
        T            : SNN time steps
        tau          : LIF membrane time constant
        levels       : number of transformer blocks
        d_model      : attention / embedding dimension
        n_heads      : number of attention heads
        d_ff         : feedforward hidden dim (default 4 × d_model)
        common_thr   : LIF spike threshold (default 1.0)
        qk_scale     : attention score scale (default 0.125)
        dropout      : dropout on the global embedding (default 0.1)
        encoder_type : 'conv' | 'delta' | 'repeat' (default 'conv')
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
        enc_type = getattr(configs, 'encoder_type', 'conv')

        self.pred_len = pred_len
        self.D = D

        self.temporal_encoder = _ENCODERS[enc_type](T, tau)

        self.encoder = nn.Linear(D, d_model)
        self.init_bn = nn.BatchNorm1d(d_model)
        self.init_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr, backend='torch',
        )

        self.blocks = nn.ModuleList([
            Block(length=seq_len, tau=tau, common_thr=common_thr,
                  dim=d_model, d_ff=d_ff, heads=n_heads, qk_scale=qk_scale)
            for _ in range(depths)
        ])

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, pred_len * D)

        self._init_weights()

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

        B = x.shape[0]

        x = self.temporal_encoder(x)          # (T, B, D, L)
        x = x.transpose(-2, -1)               # (T, B, L, D)
        T, _, L, _ = x.shape

        # Project D → d_model with BN + LIF
        x_flat = x.flatten(0, 1)                                             # (TB, L, D)
        x_flat = self.encoder(x_flat)                                        # (TB, L, d_model)
        x_flat = self.init_bn(x_flat.transpose(-1, -2)).transpose(-1, -2)   # (TB, L, d_model)
        x = x_flat.reshape(T, B, L, -1).contiguous()
        x = self.init_lif(x)

        for blk in self.blocks:
            x = blk(x)                        # (T, B, L, d_model)

        x = x.mean(0)                         # (B, L, d_model)  — average over T
        x = self.dropout(x)
        x = x.mean(1)                         # (B, d_model)     — global avg over L

        x = self.out_proj(x)                  # (B, pred_len * D)
        x = x.reshape(B, self.pred_len, self.D)

        return x * std + mean
