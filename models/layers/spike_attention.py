"""
Spiking Self-Attention (SSA) blocks — adapted from SeqSNN (Microsoft, MIT License)
and TS-LIF (arXiv:2503.05108).

Copyright (c) Microsoft Corporation.
https://github.com/microsoft/SeqSNN  (SeqSNN/module/spike_attention.py)

Classes (spikingjelly LIFNode, step_mode='m'):
  SSA          — standard dot-product spiking attention (qk_scale fixed)
  SSA_XNOR     — XNOR spiking attention with learnable scale; supports three
                  positional-bias modes via attn_pe:
                    'none'  — base XNOR 
                    'gray'  — Gray-code Q/K augmentation
                    'log'   — log-distance symmetric bias 
  MLP          — spiking two-layer MLP
  Block        — SSA + MLP residual block; accepts attn_cls kwarg to swap in SSA_XNOR
  SpikingSSA, SpikingSSA_XNOR, SpikingMLP, SpikingBlock — pre-LIF Spikingformer blocks

Classes (TSLIFNode, two-compartment dual-spike):
  TSSSA        — SSA with TSLIFNode replacing all LIFNodes
  TSMLP        — MLP with TSLIFNode replacing all LIFNodes
  TSBlock      — TSSSA + TSMLP residual block

All input/output tensors have shape (T, B, L, D).
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

from models.layers.positional_encoding import (
    generate_gray_code_matrix, create_symmetric_matrix, CPGLinear,
)
from models.layers.tslif import TSLIFNode

_DETACH = True
_BACKEND = 'torch'


class SSA(nn.Module):
    """Spiking Self-Attention. Input: (T, B, L, D) -> Output: (T, B, L, D)."""

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
    """Spiking MLP. Input: (T, B, L, D) -> Output: (T, B, L, D)."""

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


class MLPCPG(nn.Module):
    """Spiking MLP with CPGLinear for fc1/fc2 — used by the CPG Spikformer variant.
    Input/output: (T, B, L, D).
    """

    def __init__(self, length, tau, common_thr, in_features, hidden_features,
                 out_features=None, num_pe_neuron=10, neuron_pe_scale=1000.0):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        def _lif():
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=common_thr, backend=_BACKEND)

        self.fc1 = CPGLinear(in_features, hidden_features,
                             num_pe_neuron=num_pe_neuron, w_max=neuron_pe_scale)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = _lif()

        self.fc2 = CPGLinear(hidden_features, out_features,
                             num_pe_neuron=num_pe_neuron, w_max=neuron_pe_scale)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = _lif()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = x.permute(1, 0, 2, 3).flatten(1, 2)                        # (B, TL, D)
        x = self.fc1(x)                                                  # (B, TL, hidden)
        x = (self.bn1(x.transpose(-1, -2)).transpose(-1, -2)
             .reshape(B, T, L, self.hidden_features).contiguous())      # (B, T, L, H)
        x = self.lif1(x.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)      # (B, T, L, H)
        x = x.flatten(1, 2)                                             # (B, TL, H)
        x = self.fc2(x)                                                  # (B, TL, out)
        x = (self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
             .reshape(B, T, L, self.out_features).contiguous())         # (B, T, L, out)
        return self.lif2(x.permute(1, 0, 2, 3))                         # (T, B, L, out)


class SSA_XNOR(nn.Module):
    """XNOR Spiking Self-Attention with learnable scale.

    Replaces the fixed dot-product similarity Q·Kᵀ with the XNOR similarity
    metric (equivalent to counting matching binary bits):
        sim(q, k) = D - Σq - Σk + 2·(q·k)

    Three positional-bias modes (attn_pe):
      'none'  — base XNOR;
      'gray'  — Q and K are augmented with Gray-coded position bits before
                XNOR. Wider feature space encodes sequence position inside the
                similarity kernel.
      'log'   — log-distance symmetric matrix added to the attention map before
                sigmoid scaling; nearby positions get higher bias.

    Assertion: attn_pe must be 'none', 'gray', or 'log'.
    Input/output: (T, B, L, D).
    """

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125,
                 attn_pe: str = 'none', gray_bits: int = 10):
        super().__init__()
        assert attn_pe in ('none', 'gray', 'log'), \
            f"attn_pe must be 'none', 'gray', or 'log', got {attn_pe!r}"
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.heads = heads
        self.attn_pe = attn_pe
        self.gray_bits = gray_bits
        scale_init = -2.0 if attn_pe == 'none' else -4.0
        self.scale = nn.Parameter(torch.tensor(scale_init))

        def _lif(thr=common_thr):
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=thr, backend=_BACKEND)

        self.q_m = nn.Linear(dim, dim);  self.q_bn = nn.BatchNorm1d(dim);  self.q_lif = _lif()
        self.k_m = nn.Linear(dim, dim);  self.k_bn = nn.BatchNorm1d(dim);  self.k_lif = _lif()
        self.v_m = nn.Linear(dim, dim);  self.v_bn = nn.BatchNorm1d(dim);  self.v_lif = _lif()
        self.attn_lif  = _lif(common_thr / 2)
        self.last_m    = nn.Linear(dim, dim)
        self.last_bn   = nn.BatchNorm1d(dim)
        self.last_lif  = _lif()

    def _proj(self, lin, bn, lif, tb, T, B, L, D):
        out = lin(tb)
        out = bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        return lif(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        tb = x.flatten(0, 1)

        q = self._proj(self.q_m, self.q_bn, self.q_lif, tb, T, B, L, D)
        k = self._proj(self.k_m, self.k_bn, self.k_lif, tb, T, B, L, D)
        v = self._proj(self.v_m, self.v_bn, self.v_lif, tb, T, B, L, D)

        head_dim = D // self.heads
        # (T, B, H, L, head_dim)
        q = q.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()

        if self.attn_pe == 'gray':
            q = generate_gray_code_matrix(q, self.gray_bits)  # (T,B,H,L, head_dim+bits)
            k = generate_gray_code_matrix(k, self.gray_bits)


        D_new = q.size(-1)
        qk = torch.matmul(q, k.transpose(-1, -2))              # (T,B,H,L,L)
        sum_q = q.sum(-1, keepdim=True)                         # (T,B,H,L,1)
        sum_k = k.sum(-1).unsqueeze(-2)                         # (T,B,H,1,L)
        attn = D_new - sum_q - sum_k + 2 * qk                  # (T,B,H,L,L)

        sig = torch.sigmoid(self.scale)
        if self.attn_pe == 'none':
            attn = (attn - attn.min()) * sig   # global min scalar, matches reference
        elif self.attn_pe == 'gray':
            attn = attn * sig
        else:  # log
            bias = create_symmetric_matrix(L, device=x.device).float()  # (1,1,1,L,L)
            attn = (attn + bias) * sig

        x = (attn @ v).transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x


class Block(nn.Module):
    """SSA + MLP residual block. Input/output: (T, B, L, D).

    Args:
        attn_cls:    Attention class to use (default: SSA). Pass SSA_XNOR for
                     XNOR variants.
        mlp_cls:     MLP class to use (default: MLP). Pass MLPCPG for CPG variant.
        mlp_kwargs:  Extra kwargs forwarded to mlp_cls (e.g. num_pe_neuron for MLPCPG).
    """

    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qk_scale=0.125,
                 attn_cls=None, mlp_cls=None, mlp_kwargs=None, **attn_kwargs):
        super().__init__()
        attn_cls = attn_cls or SSA
        mlp_cls = mlp_cls or MLP
        mlp_kwargs = mlp_kwargs or {}
        self.attn = attn_cls(length=length, tau=tau, common_thr=common_thr,
                             dim=dim, heads=heads, qk_scale=qk_scale, **attn_kwargs)
        self.mlp = mlp_cls(length=length, tau=tau, common_thr=common_thr,
                           in_features=dim, hidden_features=d_ff, **mlp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class SpikingSSA(nn.Module):
    """Spikingformer spiking self-attention. Input/output: (T, B, L, D).
    """

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qk_scale = qk_scale

        def _lif(thr=common_thr):
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=thr, backend=_BACKEND)

        self.pre_lif = _lif()   # applied to input x before QKV

        self.q_m = nn.Linear(dim, dim);  self.q_bn = nn.BatchNorm1d(dim);  self.q_lif = _lif()
        self.k_m = nn.Linear(dim, dim);  self.k_bn = nn.BatchNorm1d(dim);  self.k_lif = _lif()
        self.v_m = nn.Linear(dim, dim);  self.v_bn = nn.BatchNorm1d(dim);  self.v_lif = _lif()
        self.attn_lif = _lif(0.5)

        self.last_m  = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        # No last_lif — next block's pre_lif acts as the output gate

    def _proj(self, lin, bn, lif, tb, T, B, L, D):
        out = lin(tb)
        out = bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        return lif(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = self.pre_lif(x)                                            # pre-gate input
        tb = x.flatten(0, 1)

        q = self._proj(self.q_m, self.q_bn, self.q_lif, tb, T, B, L, D)
        k = self._proj(self.k_m, self.k_bn, self.k_lif, tb, T, B, L, D)
        v = self._proj(self.v_m, self.v_bn, self.v_lif, tb, T, B, L, D)

        head_dim = D // self.heads
        q = q.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = (attn @ v).transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)
        return x


class SpikingSSA_XNOR(nn.Module):
    """Spikingformer XNOR spiking self-attention. Input/output: (T, B, L, D).

    Combines SpikingSSA's pre-LIF gate with SSA_XNOR's similarity computation.
    attn_pe: 'gray' — Gray-code Q/K augmentation 
             'log'  — log-distance symmetric bias
    """

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125,
                 attn_pe: str = 'gray', gray_bits: int = 10):
        super().__init__()
        assert attn_pe in ('gray', 'log'), \
            f"SpikingSSA_XNOR attn_pe must be 'gray' or 'log', got {attn_pe!r}"
        assert dim % heads == 0
        self.heads = heads
        self.attn_pe = attn_pe
        self.gray_bits = gray_bits
        self.scale = nn.Parameter(torch.tensor(-4.0))

        def _lif(thr=common_thr):
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=thr, backend=_BACKEND)

        self.pre_lif = _lif()
        self.q_m = nn.Linear(dim, dim);  self.q_bn = nn.BatchNorm1d(dim);  self.q_lif = _lif()
        self.k_m = nn.Linear(dim, dim);  self.k_bn = nn.BatchNorm1d(dim);  self.k_lif = _lif()
        self.v_m = nn.Linear(dim, dim);  self.v_bn = nn.BatchNorm1d(dim);  self.v_lif = _lif()
        self.attn_lif = _lif(0.5)
        self.last_m  = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)

    def _proj(self, lin, bn, lif, tb, T, B, L, D):
        out = lin(tb)
        out = bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        return lif(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = self.pre_lif(x)
        tb = x.flatten(0, 1)

        q = self._proj(self.q_m, self.q_bn, self.q_lif, tb, T, B, L, D)
        k = self._proj(self.k_m, self.k_bn, self.k_lif, tb, T, B, L, D)
        v = self._proj(self.v_m, self.v_bn, self.v_lif, tb, T, B, L, D)

        head_dim = D // self.heads
        q = q.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()

        if self.attn_pe == 'gray':
            q = generate_gray_code_matrix(q, self.gray_bits)
            k = generate_gray_code_matrix(k, self.gray_bits)

        D_new = q.size(-1)
        qk = torch.matmul(q, k.transpose(-1, -2))
        sum_q = q.sum(-1, keepdim=True)
        sum_k = k.sum(-1).unsqueeze(-2)
        attn = D_new - sum_q - sum_k + 2 * qk

        sig = torch.sigmoid(self.scale)
        if self.attn_pe == 'gray':
            attn = attn * sig
        else:  # log
            bias = create_symmetric_matrix(L, device=x.device).float()
            attn = (attn + bias) * sig

        x = (attn @ v).transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)
        return x


class SpikingMLP(nn.Module):
    """Spikingformer MLP with pre-LIF gates. Input/output: (T, B, L, D).
    """

    def __init__(self, length, tau, common_thr, in_features, hidden_features,
                 out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        def _lif():
            return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                  surrogate_function=surrogate.ATan(),
                                  v_threshold=common_thr, backend=_BACKEND)

        self.lif1 = _lif()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.bn1  = nn.BatchNorm1d(hidden_features)

        self.lif2 = _lif()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.bn2  = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.fc1(x)
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(
            T, B, L, self.hidden_features).contiguous()
        x = self.lif2(x)
        x = x.flatten(0, 1)
        x = self.fc2(x)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(
            T, B, L, self.out_features).contiguous()
        return x


class SpikingBlock(nn.Module):
    """SpikingSSA/SpikingSSA_XNOR + SpikingMLP residual block. Input/output: (T, B, L, D).

    Args:
        attn_cls:    SpikingSSA (default) or SpikingSSA_XNOR for XNOR variants.
    """

    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qk_scale=0.125,
                 attn_cls=None, **attn_kwargs):
        super().__init__()
        attn_cls = attn_cls or SpikingSSA
        self.attn = attn_cls(length=length, tau=tau, common_thr=common_thr,
                             dim=dim, heads=heads, qk_scale=qk_scale, **attn_kwargs)
        self.mlp  = SpikingMLP(length=length, tau=tau, common_thr=common_thr,
                               in_features=dim, hidden_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class TSSSA(nn.Module):
    """TS-LIF Spiking Self-Attention. Input/output: (T, B, L, D).
    """

    def __init__(self, length, tau, common_thr, dim, heads=8, qk_scale=0.125):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_tslif = TSLIFNode()

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_tslif = TSLIFNode()

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_tslif = TSLIFNode()

        self.attn_tslif = TSLIFNode(v_threshold=0.5)

        self.last_m  = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_tslif = TSLIFNode()

    def _proj(self, lin, bn, tslif, tb, T, B, L, D):
        out = lin(tb)
        out = bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        return tslif(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        tb = x.flatten(0, 1)                                        # (TB, L, D)

        q = self._proj(self.q_m, self.q_bn, self.q_tslif, tb, T, B, L, D)
        k = self._proj(self.k_m, self.k_bn, self.k_tslif, tb, T, B, L, D)
        v = self._proj(self.v_m, self.v_bn, self.v_tslif, tb, T, B, L, D)

        head_dim = D // self.heads
        q = q.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        k = k.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        v = v.reshape(T, B, L, self.heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale            # (T,B,H,L,L)
        x = (attn @ v).transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_tslif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_tslif(x.reshape(T, B, L, D).contiguous())
        return x


class TSMLP(nn.Module):
    """TS-LIF MLP. Input/output: (T, B, L, D).

    Identical structure to MLP but every LIFNode is replaced by TSLIFNode.
    """

    def __init__(self, length, tau, common_thr, in_features, hidden_features,
                 out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1    = nn.Linear(in_features, hidden_features)
        self.bn1    = nn.BatchNorm1d(hidden_features)
        self.tslif1 = TSLIFNode()

        self.fc2    = nn.Linear(hidden_features, out_features)
        self.bn2    = nn.BatchNorm1d(out_features)
        self.tslif2 = TSLIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.fc1(x)
        x = (self.bn1(x.transpose(-1, -2)).transpose(-1, -2)
             .reshape(T, B, L, self.hidden_features).contiguous())
        x = self.tslif1(x)
        x = x.flatten(0, 1)
        x = self.fc2(x)
        x = (self.bn2(x.transpose(-1, -2)).transpose(-1, -2)
             .reshape(T, B, L, self.out_features).contiguous())
        return self.tslif2(x)


class TSBlock(nn.Module):
    """TSSSA + TSMLP residual block. Input/output: (T, B, L, D).

    TS-LIF drop-in replacement for Block: uses two-compartment dual-spike
    TSLIFNode in both attention and MLP sub-layers.
    """

    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8, qk_scale=0.125):
        super().__init__()
        self.attn = TSSSA(length=length, tau=tau, common_thr=common_thr,
                          dim=dim, heads=heads, qk_scale=qk_scale)
        self.mlp  = TSMLP(length=length, tau=tau, common_thr=common_thr,
                          in_features=dim, hidden_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
