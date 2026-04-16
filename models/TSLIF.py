"""
TS-LIF: Temporal Spiking LIF Network for long-term time series forecasting.

Inspired by TS-LIF (ICLR 2025):
  "TS-LIF: A Temporal Spiking LIF Model for Time Series Forecasting"
  https://github.com/kkking-kk/TS-LIF

Key innovations implemented here:
  1. Multi-scale Temporal LIF (T-LIF): neurons with different time constants (τ_fast,
     τ_slow) capture both local and global temporal patterns simultaneously.
  2. Learnable inter-scale fusion gate: combines fast and slow spike streams.
  3. Patch-based encoding (shared with SpikF) for efficient multi-scale processing.

Architecture:
    Input (B, L, D)
      ↓  instance norm
    PatchEncoder → (T, B, patch_dim, patch_num, D)
      ↓  N × TSLIFBlock
         ├─ fast-LIF branch (small τ)
         ├─ slow-LIF branch (large τ)
         └─ learnable gate fusion
      ↓  flatten + dense head
    Output (B, pred_len, D)
"""

import torch
import torch.nn.functional as F
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


# ---------------------------------------------------------------------------
# Patch encoder (same design as in SpikF / iSpikformer)
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, tau, D):
        super().__init__()
        self.patch_projector = nn.Linear(input_len // patch_num, patch_dim)
        self.bn = nn.BatchNorm2d(patch_dim)
        self.encoder_lif = MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch')

        self.T = T
        self.patch_dim = patch_dim
        self.patch_num = patch_num

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x = x.view(B, self.patch_num, L // self.patch_num, D).contiguous()
        x = x.transpose(-1, -2).contiguous()          # (B, patch_num, D, L//patch_num)
        x = self.patch_projector(x)                    # (B, patch_num, D, patch_dim)
        x = x.repeat(self.T, 1, 1, 1, 1)              # (T, B, patch_num, D, patch_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()     # (T, B, patch_dim, patch_num, D)
        x = x.flatten(0, 1)                            # (T*B, patch_dim, patch_num, D)
        x = self.bn(x)
        x = x.view(self.T, B, self.patch_dim, self.patch_num, D)
        x = self.encoder_lif(x)                        # (T, B, patch_dim, patch_num, D)
        return x


# ---------------------------------------------------------------------------
# Multi-scale T-LIF block
# ---------------------------------------------------------------------------

class TSLIFBlock(nn.Module):
    """
    One TS-LIF block with a fast and a slow LIF branch.

    Both branches receive the same input spikes from the previous block.
    Their outputs are fused via a learnable sigmoid gate — the model learns
    how much short-term vs. long-term temporal context to retain.
    """

    def __init__(self, patch_num, patch_dim, D, tau_fast, tau_slow):
        super().__init__()
        # Fast branch: small τ → reacts quickly, forgets quickly
        self.fast_lif = MultiStepLIFNode(tau=tau_fast, detach_reset=True, backend='torch')
        # Slow branch: large τ → integrates over many steps, retains long-range info
        self.slow_lif = MultiStepLIFNode(tau=tau_slow, detach_reset=True, backend='torch')

        # Learnable gate per (patch_dim, patch_num, D) position
        self.gate = nn.Linear(2 * patch_dim, patch_dim)

        self.bn_fast = nn.BatchNorm2d(patch_dim)
        self.bn_slow = nn.BatchNorm2d(patch_dim)
        self.bn_out = nn.BatchNorm2d(patch_dim)

    def forward(self, x):
        # x: (T, B, patch_dim, patch_num, D)
        T, B, pd, pn, D = x.shape

        flat = x.flatten(0, 1)      # (TB, pd, pn, D)

        fast_in = self.bn_fast(flat).view(T, B, pd, pn, D)
        slow_in = self.bn_slow(flat).view(T, B, pd, pn, D)

        fast_spk = self.fast_lif(fast_in)   # (T, B, pd, pn, D)
        slow_spk = self.slow_lif(slow_in)   # (T, B, pd, pn, D)

        # Gate fusion: decide per position how much fast/slow to mix
        # Concatenate along patch_dim axis, then project back
        concat = torch.cat([fast_spk, slow_spk], dim=2)  # (T, B, 2*pd, pn, D)
        concat = concat.permute(0, 1, 3, 4, 2)           # (T, B, pn, D, 2*pd)
        gate = torch.sigmoid(self.gate(concat))           # (T, B, pn, D, pd)
        gate = gate.permute(0, 1, 4, 2, 3)               # (T, B, pd, pn, D)

        fused = gate * fast_spk + (1 - gate) * slow_spk  # (T, B, pd, pn, D)

        # Residual + batch-norm
        out = fused + x
        out_flat = out.flatten(0, 1)
        out_flat = self.bn_out(out_flat)
        return out_flat.view(T, B, pd, pn, D)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    TS-LIF: multi-scale Temporal Spiking LIF forecaster.

    Args (from configs namespace):
        seq_len    : input length L
        pred_len   : forecast horizon H
        enc_in     : number of input variables D
        patch_num  : number of patches (must divide seq_len)
        patch_dim  : patch embedding dimension
        T          : SNN time-steps
        levels     : number of TSLIFBlock layers
        tau        : base LIF time constant (fast branch uses tau,
                     slow branch uses 2*tau for richer long-range memory)
        hidden_dim : size of the intermediate dense projection
    """

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        D = configs.enc_in
        patch_num = configs.patch_num
        patch_dim = configs.patch_dim
        T = configs.T
        levels = configs.levels
        tau = configs.tau
        hidden_dim = configs.hidden_dim

        tau_fast = max(1.01, tau)           # fast branch
        tau_slow = max(1.01, tau * 2.0)    # slow branch: longer memory

        self.encoder = PatchEncoder(seq_len, patch_num, patch_dim, T, tau, D)

        self.blocks = nn.ModuleList([
            TSLIFBlock(patch_num, patch_dim, D, tau_fast, tau_slow)
            for _ in range(levels)
        ])

        self.dense1 = nn.Linear(patch_num * patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = nn.GELU()

    def forward(self, x):
        # x: (B, L, D)
        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        x = self.encoder(x)              # (T, B, patch_dim, patch_num, D)
        for blk in self.blocks:
            x = blk(x)

        T, B, pd, pn, D = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()   # (T, B, D, pd, pn)
        x = x.flatten(-2, -1)                         # (T, B, D, pd*pn)
        x = self.dense1(x)                            # (T, B, D, hidden_dim)
        x = x.flatten(0, 1)                           # (T*B, D, hidden_dim)

        # BatchNorm1d on D "channels" (treats hidden_dim as sequence length)
        x = x.transpose(1, 2)                         # (T*B, hidden_dim, D)
        x = self.bn(x.reshape(T * B * x.size(1), D).T).T  # norm per variable
        x = x.reshape(T * B, x.size(1), D).transpose(1, 2)

        x = self.activ(x)                             # (T*B, D, hidden_dim)
        x = self.dense2(x)                            # (T*B, D, pred_len)
        x = x.reshape(T, B, D, -1)

        x = x * std.unsqueeze(0) + mean.unsqueeze(0)
        # Average over T: (B, D, pred_len) → (B, pred_len, D)
        return x.mean(0).permute(0, 2, 1).contiguous()
