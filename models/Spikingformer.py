"""Spikingformer — pre-LIF spiking transformer with four attention variants.

Combines four SeqSNN-RPE Spikingformer variants into a single class via `attn_type`:
  'standard'  — standard pre-LIF SSA (dot-product)
  'xnor'      — XNOR attention, global min-shift; pe_type='conv' required
  'xnor_gray' — XNOR + Gray-code Q/K augmentation; pe_type='conv' required
  'xnor_log'  — XNOR + log-distance integer bias; pe_type='conv' required


Architecture:
    (B, L, D)
      -> RevIN norm
      -> SpikeEncoder                      (T, B, D, L)
      -> transpose                         (T, B, L, D)
      -> [ConvPE if pe_type='conv']        (T, B, L, D)
      -> Linear(D -> d_model) + init_bn    (T, B, L, d_model)  [no init_lif]
      -> SpikingBlock x blocks             (T, B, L, d_model)
          standard: SpikingSSA + SpikingMLP
          xnor:     SpikingSSA_XNOR(attn_pe='none', scale -2) + SpikingMLP  [not in reference, added for completeness]
          xnor_gray:SpikingSSA_XNOR(attn_pe='gray', scale -4) + SpikingMLP
          xnor_log: SpikingSSA_XNOR(attn_pe='log',  scale -4) + SpikingMLP
      -> mean(T)                           (B, L, d_model)
      -> dropout
      -> mean(L)                           (B, d_model)
      -> Linear(d_model -> pred_lenxD) -> reshape
      -> denorm
    (B, pred_len, D)

References:
    SeqSNN (Microsoft, MIT License): https://github.com/microsoft/SeqSNN
      SeqSNN-RPE/network/snn/spikingformer.py
      SeqSNN-RPE/network/snn/spikingformer_xnor_gray.py
      SeqSNN-RPE/network/snn/spikingformer_xnor_log.py
"""

import torch
from torch import nn
from spikingjelly.activation_based import functional

from models.layers.spike_attention import SpikingBlock, SpikingSSA_XNOR
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding

_XNOR_TYPES = ('xnor', 'xnor_gray', 'xnor_log')
_ATTN_PE_MAP = {'xnor': 'none', 'xnor_gray': 'gray', 'xnor_log': 'log'}
_XNOR_SCALE  = {'xnor': -2.0, 'xnor_gray': -4.0, 'xnor_log': -4.0}


class Spikingformer(nn.Module):
    """Spikingformer: pre-LIF spiking transformer for long-term forecasting.

    Args:
        input_len:       Sequence length (L = seq_len).
        T:               Number of SNN time steps.
        blocks:          Number of SpikingBlock layers.
        D:               Number of input/output channels (enc_in).
        pred_len:        Prediction horizon.
        tau:             LIF membrane time constant.
        d_model:         Embedding and attention dimension (must be divisible by heads).
        d_ff:            Feedforward hidden dim (default: 4 x d_model).
        heads:           Attention heads (default 8).
        common_thr:      LIF firing threshold (default 1.0).
        qk_scale:        Q·K scale for standard SSA (default 0.125; ignored for XNOR).
        encoder_type:    Spike encoder — 'conv' or 'delta'.
        pe_type:         Positional encoding — 'none' or 'conv' (XNOR requires 'conv').
        attn_type:       Attention variant — 'standard', 'xnor', 'xnor_gray', 'xnor_log'.
        gray_bits:       Gray-code bits for 'xnor_gray' (default 10).
        dropout:         Dropout before out_proj (default 0.1).
        normalize:       RevIN-style normalization (default True).
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
            f"attn_type must be 'standard', 'xnor', 'xnor_gray', or 'xnor_log', "
            f"got {attn_type!r}"
        )
        if attn_type in _XNOR_TYPES:
            assert pe_type == 'conv', (
                f"Spikingformer XNOR variants require pe_type='conv', got {pe_type!r}"
            )

        self.T = T
        self.D = D
        self.pred_len = pred_len
        self.attn_type = attn_type
        self.pe_type = pe_type
        self.normalize = normalize
        d_ff = d_ff or d_model * 4

        # Spike encoder: (B, L, D) -> (T, B, D, L)
        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        # Positional encoding (conv for XNOR, optional for standard)
        if pe_type != 'none':
            self.pe = PositionEmbedding(
                input_size=D, pe_type=pe_type, pe_mode='add', num_steps=T,
            )

        # Input embedding: Linear + init_bn (no init_lif — pre_lif inside each block)
        self.input_embed = nn.Linear(D, d_model)
        self.init_bn = nn.BatchNorm1d(d_model)

        # Spikingformer blocks
        if attn_type in _XNOR_TYPES:
            self.transformer_blocks = nn.ModuleList([
                SpikingBlock(length=input_len, tau=tau, common_thr=common_thr,
                             dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale,
                             attn_cls=SpikingSSA_XNOR,
                             attn_pe=_ATTN_PE_MAP[attn_type],
                             gray_bits=gray_bits)
                for _ in range(blocks)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                SpikingBlock(length=input_len, tau=tau, common_thr=common_thr,
                             dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale)
                for _ in range(blocks)
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
        if self.normalize:
            mean = x.mean(1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / std

        functional.reset_net(self)

        B = x.shape[0]

        x = self.spike_encoder(x)                                      # (T, B, D, L)
        x = x.transpose(-2, -1)                                        # (T, B, L, D)
        T, _, L, _ = x.shape

        if self.pe_type != 'none':
            x = self.pe(x)                                             # (T, B, L, D)

        # Embedding: Linear + init_bn (no init_lif)
        x = self.input_embed(x.flatten(0, 1))                          # (TB, L, d_model)
        x = self.init_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.reshape(T, B, L, -1).contiguous()                       # (T, B, L, d_model)

        for blk in self.transformer_blocks:
            x = blk(x)                                                 # (T, B, L, d_model)

        x = x.mean(0)                                                  # (B, L, d_model)
        x = self.dropout(x)
        x = x.mean(1)                                                  # (B, d_model)

        x = self.out_proj(x)                                           # (B, pred_len * D)
        x = x.reshape(B, self.pred_len, self.D)

        if self.normalize:
            x = x * std + mean

        return x
