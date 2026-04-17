"""iSpikformer model implementation.

Reference:
    Lv, Changze, Yansen Wang, Dongqi Han, Xiaoqing Zheng,
    Xuanjing Huang, and Dongsheng Li.
    "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks."
    arXiv preprint arXiv:2402.01533, 2024.
    Paper: https://arxiv.org/abs/2402.01533

Official project repository:
    https://github.com/microsoft/SeqSNN

Architecture follows SeqSNN's iSpikformer network (iTransformer paradigm):
  1. Spike encoder (conv or delta)  → (T, B, D, L)
  2. Transpose                      → (T, B, L, D)
  3. DataEmbeddingInverted          → (T, B, D, d_model)
     Each of the D channels is embedded over its L-step history: Linear(L, d_model)
  4. Stack of Block(SSA + MLP)      → (T, B, D, d_model)
     Self-attention runs across the D channel tokens (multivariate correlation)
  5. Last step                      → (B, D, d_model)
  6. Linear(d_model, pred_len)      → (B, D, pred_len) → transpose → (B, pred_len, D)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.spike_attention import Block


def _make_lif(tau: float, common_thr: float = 1.0) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=tau, step_mode='m', detach_reset=True,
        surrogate_function=surrogate.ATan(),
        v_threshold=common_thr, backend='torch',
    )


class DataEmbeddingInverted(nn.Module):
    """Inverted channel-wise temporal embedding.

    Maps each channel's L-step time series to a d_model-dim token via a
    linear projection, BN, and LIF spike layer.  Follows the iTransformer
    principle: treat variates as tokens rather than time steps.

    Input : (T, B, L, D) — L time steps, D channels
    Output: (T, B, D, d_model) — D channel tokens, each with d_model features
    """

    def __init__(self, seq_len: int, d_model: int, tau: float, common_thr: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = _make_lif(tau, common_thr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D)
        T, B, L, D = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)                        # (TB, D, L)
        x = self.value_embedding(x)                                     # (TB, D, d_model)
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)             # (TB, D, d_model)
        x = x.reshape(T, B, D, self.d_model)
        return self.lif(x)                                              # (T, B, D, d_model)


class iSpikformer(nn.Module):
    """iSpikformer: inverted spiking transformer for long-term time-series forecasting.

    Args:
        input_len:    Sequence length (L = seq_len).
        T:            Number of SNN time steps.
        blocks:       Number of transformer blocks (depth).
        D:            Number of input/output channels (enc_in).
        pred_len:     Prediction horizon.
        tau:          LIF membrane time constant.
        d_model:      Per-channel embedding dimension (mapped from ``alpha`` in args).
        d_ff:         Feedforward hidden size in MLP blocks (default: d_model * 4).
        heads:        Number of attention heads (must divide d_model).
        common_thr:   LIF firing threshold for all neurons.
        qk_scale:     Q·K scaling factor in SSA (default 0.125 = 1/8).
        encoder_type: Spike encoder variant — ``'conv'`` or ``'delta'``.
        normalize:    RevIN-style per-instance normalization.
    """

    def __init__(
        self,
        input_len: int,
        T: int,
        blocks: int,
        D: int,
        pred_len: int,
        tau: float,
        d_model: int,
        d_ff: int = None,
        heads: int = 8,
        common_thr: float = 1.0,
        qk_scale: float = 0.125,
        encoder_type: str = 'conv',
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        d_ff = d_ff or d_model * 4

        # Spike encoder: (B, L, D) → (T, B, D, L)
        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        # Inverted embedding: (T, B, L, D) → (T, B, D, d_model)
        self.emb = DataEmbeddingInverted(input_len, d_model, tau, common_thr)

        # Transformer blocks: SSA + MLP over D channel tokens with d_model features.
        # Block(SSA+MLP) operates on (T, B, seq, dim); here seq=D, dim=d_model.
        self.attn_blocks = nn.ModuleList([
            Block(
                length=D,
                tau=tau,
                common_thr=common_thr,
                dim=d_model,
                d_ff=d_ff,
                heads=heads,
                qk_scale=qk_scale,
            )
            for _ in range(blocks)
        ])

        # Decoder: project each channel's d_model embedding to pred_len
        self.dense = nn.Linear(d_model, pred_len)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()                # (B, 1, D)
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()                                                  # (B, 1, D)
            x = x / std

        # Spike encoding
        h = self.spike_encoder(x)                                      # (T, B, D, L)
        h = h.transpose(-1, -2)                                        # (T, B, L, D)

        # Inverted embedding: embed each channel over its temporal history
        h = self.emb(h)                                                # (T, B, D, d_model)

        # Self-attention across channel tokens
        for blk in self.attn_blocks:
            h = blk(h)                                                 # (T, B, D, d_model)

        # Decode: last step→ project d_model → pred_len
        out = h[-1, :, :, :]                                          # (B, D, d_model)
        out = self.dense(out)                                          # (B, D, pred_len)
        out = out.transpose(-1, -2)                                    # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean                                     # broadcast (B, 1, D)

        return out
