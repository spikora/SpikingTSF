"""
ITransformer — ANN baseline for long-term time series forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Copyright (c) Microsoft Corporation.
  Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks",
  ICML 2024.  https://github.com/microsoft/SeqSNN  (SeqSNN/network/ann/itransformer.py)

Which itself is based on:
  Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
  ICLR 2024.  https://github.com/thuml/iTransformer  (MIT License)

Key idea: treat each variate as a token (invert the Transformer's usual role of time-step tokens).
This makes attention operate over the D-dimensional variate axis rather than the L-dimensional
time axis, allowing the model to capture inter-variate dependencies at low cost.

Adaptation notes:
  - Per-sample instance normalization follows TSLib unified evaluation protocol
  - Output projection included inside the model: (B, pred_len, D) returned directly
  - No timestamp covariates — pure value-based embedding

Architecture:
    Input (B, L, D)
      ↓  per-sample instance norm
    DataEmbedding_inverted: (B, L, D) → (B, D, d_model)   [variates as tokens]
      ↓  N × EncoderLayer (full attention over D tokens)
    Linear(d_model → pred_len) applied per variate → (B, D, pred_len)
    Permute → (B, pred_len, D)
      ↓  denorm
    Output (B, pred_len, D)
"""

from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Attention internals
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.attn = FullAttention(dropout=dropout)

    def forward(self, x, attn_mask=None):
        B, N, _ = x.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, N, H, -1)
        k = self.k_proj(x).view(B, N, H, -1)
        v = self.v_proj(x).view(B, N, H, -1)
        out = self.attn(q, k, v, attn_mask)
        return self.out_proj(out.view(B, N, -1))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1, activation='gelu'):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(x, attn_mask))
        y = x = self.norm1(x)
        y = self.dropout(self.act(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    ITransformer: inverted attention ANN baseline.

    Args (from configs namespace):
        seq_len    : input length L
        pred_len   : forecast horizon H
        enc_in     : number of input variables D
        levels     : number of encoder layers (default 2)
        d_model    : embedding / attention dimension (default 512)
        n_heads    : attention heads (default 8)
        d_ff       : feedforward dim (default 2048)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        D = configs.enc_in
        e_layers = configs.levels
        d_model = getattr(configs, 'd_model', 512)
        n_heads = getattr(configs, 'n_heads', 8)
        d_ff = getattr(configs, 'd_ff', 2048)
        dropout = getattr(configs, 'dropout', 0.1)

        self.pred_len = pred_len

        # Inverted embedding: map L time steps to d_model per variate
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Project each variate embedding to pred_len
        self.out_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        B, L, D = x.shape

        # Invert: variates become tokens — (B, D, L) → (B, D, d_model)
        x = x.permute(0, 2, 1)          # (B, D, L)
        x = self.value_embedding(x)      # (B, D, d_model)
        x = self.emb_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)                 # (B, D, d_model)
        x = self.norm(x)

        x = self.out_proj(x)             # (B, D, pred_len)
        x = x.permute(0, 2, 1)          # (B, pred_len, D)

        return x * std + mean
