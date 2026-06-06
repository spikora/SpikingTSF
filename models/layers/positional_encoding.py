"""Positional encoding modules for SNN time-series models.

Adapted from SeqSNN (Microsoft, MIT License):
https://github.com/microsoft/SeqSNN  (SeqSNN/module/positional_encoding.py)

All main PE modules accept and return (T, B, L, D) tensors.

Additional utilities for XNOR attention variants:
  generate_gray_code_matrix  — appends Gray-code position bits to Q/K tensors
  create_symmetric_matrix    — builds log-distance symmetric bias for XNOR_Log attention
  CPGLinear                  — CPG-modulated input projection used by SpikformerV CPG variant
"""

import math
import copy

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron


def _random_binary_matrix(rows: int, cols: int) -> torch.Tensor:
    m = torch.randint(0, 2, (rows, cols))
    return torch.where(m == 0, -torch.ones_like(m), torch.ones_like(m)).float()


class NeuronPE(nn.Module):
    """Binarized sinusoidal PE via heaviside thresholding (spike-compatible).

    Applies over the flattened T*L dimension so each SNN step+position
    combination receives a unique encoding.
    """

    def __init__(self, d_model: int, pe_mode: str = 'add', num_pe_neuron: int = 10,
                 neuron_pe_scale: float = 1000.0, dropout: float = 0.1,
                 num_steps: int = 4, **kwargs):
        super().__init__()
        self.pe_mode = pe_mode
        self.dropout = nn.Dropout(p=dropout)
        n = num_pe_neuron if pe_mode == 'concat' else copy.deepcopy(d_model)
        self.num_pe_neuron = n
        max_len = 50000
        pe = torch.zeros(max_len, n)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n, 2).float() * (-math.log(neuron_pe_scale) / n)
        )
        div_term_odd = torch.exp(
            torch.arange(0, n - 1, 2).float() * (-math.log(neuron_pe_scale) / n)
        )
        pe[:, 0::2] = torch.heaviside(torch.sin(position * div_term) - 0.8, torch.tensor([1.0]))
        pe[:, 1::2] = torch.heaviside(torch.cos(position * div_term_odd) - 0.8, torch.tensor([1.0]))
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))  # (max_len, 1, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D)
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3).flatten(1, 2)  # (B, TL, D)
        TL = x.size(1)
        if self.pe_mode == 'concat':
            tmp = self.pe[:TL, :].repeat(1, B, 1).transpose(0, 1)  # (B, TL, n)
            x = torch.cat([x, tmp], dim=-1)
        else:
            x = x + self.pe[:TL, :].transpose(0, 1)  # (B, TL, D)
        x = x.transpose(0, 1).reshape(T, L, B, -1).permute(0, 2, 1, 3)  # (T, B, L, D')
        return self.dropout(x)


class RandomPE(nn.Module):
    """Random ±1 positional encoding.

    Applies over the flattened T*L dimension (same convention as NeuronPE).
    """

    def __init__(self, d_model: int, pe_mode: str = 'add', num_pe_neuron: int = 10,
                 neuron_pe_scale: float = 1000.0, dropout: float = 0.1,
                 num_steps: int = 4, **kwargs):
        super().__init__()
        self.pe_mode = pe_mode
        self.dropout = nn.Dropout(p=dropout)
        n = num_pe_neuron if pe_mode == 'concat' else copy.deepcopy(d_model)
        self.num_pe_neuron = n
        pe = _random_binary_matrix(5000, n)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))  # (5000, 1, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D)
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3).flatten(1, 2)  # (B, TL, D)
        TL = x.size(1)
        if self.pe_mode == 'concat':
            tmp = self.pe[:TL, :].repeat(1, B, 1).transpose(0, 1)
            x = torch.cat([x, tmp], dim=-1)
        else:
            x = x + self.pe[:TL, :].transpose(0, 1)
        x = x.transpose(0, 1).reshape(T, L, B, -1).permute(0, 2, 1, 3)
        return self.dropout(x)


class StaticPE(nn.Module):
    """Standard sinusoidal positional encoding (ANN-style)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        div_term_odd = torch.exp(
            torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, TB, D)  — called via PositionEmbedding after transpose
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConvPE(nn.Module):
    """Convolutional relative positional encoding with spiking neuron."""

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000, num_steps: int = 4, **kwargs):
        super().__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(
            step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(), v_threshold=1.0,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, TB, D)  — called via PositionEmbedding after transpose
        L, TB, D = x.shape
        x_feat = x.permute(1, 2, 0)                                               # (TB, D, L)
        x_feat = self.rpe_conv(x_feat)                                             # (TB, D, L)
        x_feat = self.rpe_bn(x_feat).reshape(self.T, TB // self.T, D, L).contiguous()  # (T, B, D, L)
        x_feat = self.rpe_lif(x_feat)
        x_feat = x_feat.flatten(0, 1)                                              # (TB, D, L)
        x_feat = self.dropout(x_feat).permute(2, 0, 1)                            # (L, TB, D)
        return x + x_feat


class PositionEmbedding(nn.Module):
    """Dispatcher for positional embedding types.

    Supported pe_type values:
      'none'   — no PE (identity; caller should skip calling this module)
      'learn'  — learnable embedding lookup
      'static' — standard sinusoidal PE
      'conv'   — convolutional relative PE (spiking)
      'neuron' — binarized sinusoidal PE (spike-compatible)
      'random' — random ±1 PE

    For 'neuron'/'random' with pe_mode='concat', output channel dim grows
    from d_model to d_model + num_pe_neuron; caller must widen input_proj.
    """

    def __init__(self, input_size: int, pe_type: str, max_len: int = 5000,
                 pe_mode: str = 'add', num_pe_neuron: int = 10,
                 neuron_pe_scale: float = 1000.0, dropout: float = 0.1,
                 num_steps: int = 4):
        super().__init__()
        self.pe_type = pe_type
        if pe_type == 'learn':
            self.emb = nn.Embedding(max_len, input_size)
        elif pe_type == 'static':
            self.emb = StaticPE(d_model=input_size, max_len=max_len, dropout=dropout)
        elif pe_type == 'conv':
            self.emb = ConvPE(d_model=input_size, max_len=max_len,
                              dropout=dropout, num_steps=num_steps)
        elif pe_type == 'neuron':
            self.emb = NeuronPE(d_model=input_size, pe_mode=pe_mode,
                                num_pe_neuron=num_pe_neuron,
                                neuron_pe_scale=neuron_pe_scale,
                                dropout=dropout, num_steps=num_steps)
        elif pe_type == 'random':
            self.emb = RandomPE(d_model=input_size, pe_mode=pe_mode,
                                num_pe_neuron=num_pe_neuron,
                                neuron_pe_scale=neuron_pe_scale,
                                dropout=dropout, num_steps=num_steps)
        else:
            raise ValueError(f'Unknown PE type: {pe_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D) → returns (T, B, L, D') where D'=D or D+num_pe_neuron
        if self.pe_type == 'learn':
            T, B, L, _ = x.shape
            pos = torch.arange(L, device=x.device)
            emb = self.emb(pos).unsqueeze(0).unsqueeze(0).expand(T, B, -1, -1)  # (T, B, L, D)
            return x + emb
        elif self.pe_type in ('static', 'conv'):
            T, B, L, _ = x.shape
            x_flat = x.flatten(0, 1)                          # (TB, L, D)
            x_flat = self.emb(x_flat.transpose(0, 1)).transpose(0, 1)  # (TB, L, D)
            return x_flat.reshape(T, B, L, -1)
        else:  # neuron, random
            return self.emb(x)


# ---------------------------------------------------------------------------
# XNOR attention utilities  (used by SSA_XNOR in spike_attention.py)
# ---------------------------------------------------------------------------

def generate_gray_code_matrix(M: torch.Tensor, num_bits: int = 10) -> torch.Tensor:
    """Append Gray-code position bits to Q or K tensors for XNOR_Gray attention.

    Encodes L sequence positions as unique Gray-code bit vectors and concatenates
    them to the last dimension of M, widening the feature space before XNOR
    similarity. Codes are shared across T steps (position is sequence-only).

    Args:
        M:        (T, B, H, L, D_head) — per-head Q or K tensor.
        num_bits: Number of Gray-code bits to append (default 10 → 2^10=1024 unique codes).

    Returns:
        (T, B, H, L, D_head + num_bits)
    """
    T, B, H, L, _ = M.shape
    indices = torch.arange(L, device=M.device)
    gray_codes = indices ^ (indices >> 1)
    gray_bits = (
        (gray_codes.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1, device=M.device)) & 1
    ).float()                                                        # (L, num_bits)
    gray_bits = gray_bits.view(1, 1, 1, L, num_bits).expand(T, B, H, L, num_bits)
    return torch.cat((M, gray_bits), dim=-1)


def create_symmetric_matrix(L: int, device=None) -> torch.Tensor:
    """Build a symmetric log-distance bias matrix for XNOR_Log attention.

    Entry (i, j) = ceil(log2((L-1) / (|i-j| + 1))), clamped to ≥ 0.
    Diagonal entries are the largest (distance=0 → log of (L-1)/1 = L-1),
    decaying as positions diverge. Returned with broadcast-ready leading
    singleton dims (1, 1, 1, L, L).

    Args:
        L:      Sequence length.
        device: Target torch device (optional).

    Returns:
        Tensor of shape (1, 1, 1, L, L).
    """
    if L <= 1:
        matrix = torch.zeros((L, L), dtype=torch.int)
    else:
        i = torch.arange(L).view(-1, 1)
        j = torch.arange(L).view(1, -1)
        dist = torch.abs(i - j).float()
        matrix = torch.ceil(
            torch.log2((L - 1) / (dist + 1))
        ).clamp(min=0).int()
    if device is not None:
        matrix = matrix.to(device)
    return matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)            # (1, 1, 1, L, L)


# ---------------------------------------------------------------------------
# CPGLinear — CPG-modulated input projection (used by SpikformerV 'cpg' variant)
# ---------------------------------------------------------------------------

class CPGLinear(nn.Module):
    """Position-aware input projection using Central Pattern Generator encoding.

    Combines a standard linear projection with a CPG positional bias that uses
    the same binarized sinusoidal formula as NeuronPE (heaviside-thresholded
    sin/cos) but applied as an additive linear bias over position, not as a
    pre-projection PE step.

    Replaces the (PositionEmbedding + nn.Linear) pair in the 'cpg' Spikformer
    variant: the CPG signal is fused directly into the input projection so no
    separate PE module is needed.

    Args:
        input_size:    Input channel dimension D.
        output_size:   Output embedding dimension d_model.
        num_pe_neuron: Number of CPG oscillator neurons (default 10).
        w_max:         Frequency scale (default 10000.0, same as NeuronPE).
        max_len:       Maximum supported sequence*step length T*L (default 50000).
        dropout:       Dropout on input before projection (default 0.1).

    Input:  (B, TL, D)  — B batches, TL = T*seq_len flattened positions, D channels.
    Output: (B, TL, d_model)
    """

    def __init__(self, input_size: int, output_size: int,
                 num_pe_neuron: int = 10, w_max: float = 10000.0,
                 max_len: int = 50000, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, num_pe_neuron)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, num_pe_neuron, 2).float() * (-math.log(w_max) / num_pe_neuron)
        )
        div_term_odd = torch.exp(
            torch.arange(0, num_pe_neuron - 1, 2).float() * (-math.log(w_max) / num_pe_neuron)
        )
        pe[:, 0::2] = torch.heaviside(torch.sin(position * div_term) - 0.8, torch.ones(1))
        pe[:, 1::2] = torch.heaviside(torch.cos(position * div_term_odd) - 0.8, torch.ones(1))
        self.register_buffer('cpg', pe)                             # (max_len, num_pe_neuron)

        self.inp_linear = nn.Linear(input_size, output_size)
        self.cpg_linear = nn.Linear(num_pe_neuron, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, TL, D)
        cpg_bias = self.cpg_linear(self.cpg[:x.size(1)])           # (TL, output_size)
        return self.inp_linear(self.dropout(x)) + cpg_bias         # (B, TL, output_size)
