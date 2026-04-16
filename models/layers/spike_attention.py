"""
Spiking Self-Attention (SSA) blocks — adapted from SeqSNN (Microsoft, MIT License).

Copyright (c) Microsoft Corporation.
https://github.com/microsoft/SeqSNN  (SeqSNN/module/spike_attention.py)

All neurons use spikingjelly activation_based API with step_mode='m'.
Input/output tensors have shape (T, B, L, D).
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

_DETACH = True
_BACKEND = 'torch'


class SSA(nn.Module):
    """Spiking Self-Attention. Input: (T, B, L, D) → Output: (T, B, L, D)."""

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.heads = heads
        self.qk_scale = qk_scale

        def _lif(thr=common_thr):
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=thr, backend=_BACKEND)

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = _lif()

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = _lif()

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = _lif()

        self.attn_lif = _lif(common_thr / 2)

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = _lif()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        tb = x.flatten(0, 1)  # (TB, L, D)

        def _proj(lin, bn, lif, inp):
            out = lin(inp)                                              # TB, L, D
            out = bn(out.transpose(-1, -2)).transpose(-1, -2)          # TB, L, D
            out = out.reshape(T, B, L, D).contiguous()
            return lif(out)

        q = _proj(self.q_m, self.q_bn, self.q_lif, tb)
        k = _proj(self.k_m, self.k_bn, self.k_lif, tb)
        v = _proj(self.v_m, self.v_bn, self.v_lif, tb)

        # Multi-head reshape
        head_dim = D // self.heads
        q = q.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale    # T,B,H,L,L
        x = (attn @ v).transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x


class MLP(nn.Module):
    """Spiking MLP. Input: (T, B, L, D) → Output: (T, B, L, D)."""

    def __init__(self, length, tau, common_thr, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        def _lif():
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=common_thr, backend=_BACKEND)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = _lif()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = _lif()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.fc1(x)
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, self.hidden_features).contiguous()
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.fc2(x)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, self.out_features).contiguous()
        return self.lif2(x)


class Block(nn.Module):
    """SSA + MLP residual block. Input/output: (T, B, L, D)."""

    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qk_scale=0.125):
        super().__init__()
        self.attn = SSA(length=length, tau=tau, common_thr=common_thr,
                        dim=dim, heads=heads, qk_scale=qk_scale)
        self.mlp = MLP(length=length, tau=tau, common_thr=common_thr,
                       in_features=dim, hidden_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
