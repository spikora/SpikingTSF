"""QKFormer — token-level Q/K spiking transformer with optional XNOR stage-3 attention.

Combines three SeqSNN-RPE QKFormer variants into a single class via `attn_type`:
  'standard'  — all stages use standard dot-product or Token-QK attention
  'xnor_gray' — stage-3 SSA uses XNOR + Gray-code Q/K; pe_type='conv' required
  'xnor_log'  — stage-3 SSA uses XNOR + log-distance bias; pe_type='conv' required

Architecture (3-stage pipeline):
  Stage 1 (1 block): TokenQKBlock — QK token attention (no V), binary spike attn map
  Stage 2 (1 block): TokenQKBlock — same
  Stage 3 (blocks-2): SpikingBlock — SpikingSSA or SpikingSSA_XNOR

  Token_QK_Attention mechanism:
    Q, K projections → Q summed over feature dim (→ scalar per position) →
    spike gate via attn_lif → elementwise multiply with K → project output.
    Complexity O(L·D) vs O(L²) for full attention — efficient for long sequences.

    (B, L, D)
      → RevIN norm
      → SpikeEncoder                       (T, B, D, L)
      → transpose                          (T, B, L, D)
      → [ConvPE if pe_type='conv']         (T, B, L, D)
      → Linear(D → d_model) + init_bn     (T, B, L, d_model)
      → TokenQKBlock × 2 (stages 1-2)     (T, B, L, d_model)
      → SpikingBlock × (blocks-2) (stage3)(T, B, L, d_model)
          standard:  SpikingSSA + SpikingMLP
          xnor_gray: SpikingSSA_XNOR(attn_pe='gray') + SpikingMLP
          xnor_log:  SpikingSSA_XNOR(attn_pe='log')  + SpikingMLP
      → mean(T)                            (B, L, d_model)
      → dropout → mean(L)                  (B, d_model)
      → Linear(d_model → pred_len×D)
      → denorm
    (B, pred_len, D)

References:
    SeqSNN (Microsoft, MIT License): https://github.com/microsoft/SeqSNN
      SeqSNN-RPE/network/snn/qkformer.py
      SeqSNN-RPE/network/snn/qkformer_xnor_gray.py
      SeqSNN-RPE/network/snn/qkformer_xnor_log.py
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from models.layers.spike_attention import SpikingBlock, SpikingSSA_XNOR
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding

_XNOR_TYPES = ('xnor_gray', 'xnor_log')
_ATTN_PE_MAP = {'xnor_gray': 'gray', 'xnor_log': 'log'}

_DETACH  = True
_BACKEND = 'torch'


def _make_lif(tau, common_thr=1.0):
    return neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                          surrogate_function=surrogate.ATan(),
                          v_threshold=common_thr, backend=_BACKEND)


class TokenQKAttention(nn.Module):
    """Token-level Q/K attention with no V matrix. Input/output: (T, B, L, D).

    Computes a scalar attention token per position by summing Q over the feature
    dimension, then spike-gates the result and multiplies elementwise with K.
    Complexity O(L·D) per layer — no L² attention matrix.

    Q/K BN is applied on the D dimension after transposing to (T, B, D, L),
    preserving the channel-normalisation semantics from the reference.
    """

    def __init__(self, dim: int, heads: int, tau: float, common_thr: float = 1.0):
        super().__init__()
        assert dim % heads == 0
        self.dim  = dim
        self.heads = heads

        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.q_bn     = nn.BatchNorm1d(dim)
        self.q_lif    = _make_lif(tau)

        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.k_bn     = nn.BatchNorm1d(dim)
        self.k_lif    = _make_lif(tau)

        # Threshold 0.5 to gate the summed Q token
        self.attn_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=_DETACH,
                                       surrogate_function=surrogate.ATan(),
                                       v_threshold=0.5, backend=_BACKEND)

        self.proj_linear = nn.Linear(dim, dim, bias=False)
        self.proj_bn     = nn.BatchNorm1d(dim)
        self.proj_lif    = _make_lif(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        tb = x.flatten(0, 1)                                          # (TB, L, D)

        # Q path: Linear → BN on D-axis → LIF → per-head reshape
        q = self.q_linear(tb)                                         # (TB, L, D)
        q = self.q_bn(q.transpose(-1, -2)).reshape(T, B, D, L)       # (T, B, D, L)
        q = self.q_lif(q)                                             # (T, B, D, L)
        q = q.reshape(T, B, self.heads, D // self.heads, L)          # (T, B, H, d, L)

        # K path: same structure
        k = self.k_linear(tb)
        k = self.k_bn(k.transpose(-1, -2)).reshape(T, B, D, L)
        k = self.k_lif(k)
        k = k.reshape(T, B, self.heads, D // self.heads, L)          # (T, B, H, d, L)

        # Token attention: sum Q over feature dim → (T,B,H,1,L), spike-gate, mul K
        q = torch.sum(q, dim=3, keepdim=True)                        # (T, B, H, 1, L)
        attn = self.attn_lif(q)                                      # (T, B, H, 1, L)
        x = attn * k                                                  # (T, B, H, d, L)

        # Flatten heads and project
        x = x.flatten(2, 3).flatten(0, 1)                            # (TB, D, L)
        x = x.transpose(-2, -1)                                      # (TB, L, D)
        x = self.proj_bn(self.proj_linear(x).transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(T, B, L, D)
        x = self.proj_lif(x)
        return x


class TokenQKBlock(nn.Module):
    """TokenQKAttention + SpikingMLP residual block. Input/output: (T, B, L, D)."""

    def __init__(self, dim: int, heads: int, tau: float, common_thr: float,
                 d_ff: int):
        super().__init__()
        from models.layers.spike_attention import SpikingMLP
        self.tqk = TokenQKAttention(dim=dim, heads=heads, tau=tau,
                                    common_thr=common_thr)
        self.mlp = SpikingMLP(length=0, tau=tau, common_thr=common_thr,
                              in_features=dim, hidden_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tqk(x)
        x = x + self.mlp(x)
        return x


class QKFormer(nn.Module):
    """QKFormer: token-level spiking transformer for long-term forecasting.

    Args:
        input_len:   Sequence length (L = seq_len).
        T:           Number of SNN time steps.
        blocks:      Total transformer blocks. Must be ≥ 3: 2 token-QK stages + (blocks-2) spiking stages.
        D:           Number of input/output channels (enc_in).
        pred_len:    Prediction horizon.
        tau:         LIF membrane time constant.
        d_model:     Embedding and attention dimension (must be divisible by heads).
        d_ff:        Feedforward hidden dim (default: 4 × d_model).
        heads:       Attention heads (default 8).
        common_thr:  LIF firing threshold (default 1.0).
        qk_scale:    Q·K scale for SpikingSSA in stage 3 (default 0.125; ignored for XNOR).
        encoder_type:Spike encoder — 'conv' or 'delta'.
        pe_type:     Positional encoding — 'conv' (default; required for XNOR) or 'none'.
        attn_type:   Attention variant for stage 3 — 'standard', 'xnor_gray', 'xnor_log'.
        gray_bits:   Gray-code bits for 'xnor_gray' (default 10).
        dropout:     Dropout before out_proj (default 0.1).
        normalize:   RevIN-style normalization (default True).
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
        pe_type: str = 'conv',
        attn_type: str = 'standard',
        gray_bits: int = 10,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        assert attn_type in ('standard', *_XNOR_TYPES), (
            f"attn_type must be 'standard', 'xnor_gray', or 'xnor_log', got {attn_type!r}"
        )
        assert blocks >= 3, f"QKFormer needs blocks ≥ 3 (2 token-QK + ≥1 spiking), got {blocks}"
        if attn_type in _XNOR_TYPES:
            assert pe_type == 'conv', (
                f"QKFormer XNOR variants require pe_type='conv', got {pe_type!r}"
            )

        self.T = T
        self.D = D
        self.pred_len = pred_len
        self.pe_type = pe_type
        self.normalize = normalize
        d_ff = d_ff or d_model * 4

        # Spike encoder
        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        # Positional encoding
        if pe_type != 'none':
            self.pe = PositionEmbedding(
                input_size=D, pe_type=pe_type, pe_mode='add', num_steps=T,
            )

        # Input embedding: Linear + init_bn (no init_lif, matching Spikingformer)
        self.input_embed = nn.Linear(D, d_model)
        self.init_bn = nn.BatchNorm1d(d_model)

        # Stage 1 & 2: TokenQKBlock (always standard, same across variants)
        self.stage1 = TokenQKBlock(dim=d_model, heads=heads, tau=tau,
                                   common_thr=common_thr, d_ff=d_ff)
        self.stage2 = TokenQKBlock(dim=d_model, heads=heads, tau=tau,
                                   common_thr=common_thr, d_ff=d_ff)

        # Stage 3: (blocks-2) SpikingBlocks — attention variant selected here
        if attn_type in _XNOR_TYPES:
            self.stage3 = nn.ModuleList([
                SpikingBlock(length=input_len, tau=tau, common_thr=common_thr,
                             dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale,
                             attn_cls=SpikingSSA_XNOR,
                             attn_pe=_ATTN_PE_MAP[attn_type],
                             gray_bits=gray_bits)
                for _ in range(blocks - 2)
            ])
        else:
            self.stage3 = nn.ModuleList([
                SpikingBlock(length=input_len, tau=tau, common_thr=common_thr,
                             dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale)
                for _ in range(blocks - 2)
            ])

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, pred_len * D)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.normalize:
            mean = x.mean(1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / std

        functional.reset_net(self)

        B = x.shape[0]

        x = self.spike_encoder(x)                                     # (T, B, D, L)
        x = x.transpose(-2, -1)                                       # (T, B, L, D)
        T, _, L, _ = x.shape

        if self.pe_type != 'none':
            x = self.pe(x)                                            # (T, B, L, D)

        # Input embedding + init_bn (no init_lif)
        x = self.input_embed(x.flatten(0, 1))                         # (TB, L, d_model)
        x = self.init_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.reshape(T, B, L, -1).contiguous()                      # (T, B, L, d_model)

        x = self.stage1(x)
        x = self.stage2(x)
        for blk in self.stage3:
            x = blk(x)

        x = x.mean(0)                                                 # (B, L, d_model)
        x = self.dropout(x)
        x = x.mean(1)                                                 # (B, d_model)

        x = self.out_proj(x)                                          # (B, pred_len * D)
        x = x.reshape(B, self.pred_len, self.D)

        if self.normalize:
            x = x * std + mean

        return x
