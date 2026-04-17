"""Positional encoding modules for SNN time-series models.

Adapted from SeqSNN (Microsoft, MIT License):
https://github.com/microsoft/SeqSNN  (SeqSNN/module/positional_encoding.py)

All modules accept and return (T, B, L, D) tensors.
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
