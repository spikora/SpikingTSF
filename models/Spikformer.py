"""Spikformer — unified spiking transformer with five attention/PE variants.

Combines five SeqSNN-RPE Spikformer variants into a single class via `attn_type`:
  'standard'  — standard SSA (dot-product); any pe_type / pe_mode
  'cpg'       — CPG-modulated input projection + CPGLinear MLP blocks; pe_type='none'
  'xnor'      — XNOR similarity attention, global min-shift; pe_type='conv'
  'xnor_gray' — XNOR + Gray-code Q/K augmentation; pe_type='conv'
  'xnor_log'  — XNOR + log-distance symmetric integer bias; pe_type='conv'

Assertions enforced at init:
  - XNOR variants require pe_type == 'conv'
  - CPG variant requires pe_type == 'none'



Architecture (shared across all variants):
    (B, L, D)
      -> RevIN norm
      -> SpikeEncoder                      (T, B, D, L)
      -> transpose                         (T, B, L, D)
      -> [PE if pe_type != 'none']         (T, B, L, D or D+num_pe_neuron)
      -> InputEmbed [+ BN for XNOR] + LIF (T, B, L, d_model)
          standard/xnor*: Linear(D' -> d_model)  [D' widened for concat PE]
          cpg:            CPGLinear(D -> d_model) [no BN; position fused inside]
      -> Block x blocks                    (T, B, L, d_model)
          standard/xnor*: Block(SSA or SSA_XNOR, MLP)
          cpg:            Block(SSA, MLPCPG)
      -> mean(T)                           (B, L, d_model)
      -> dropout
      -> mean(L)                           (B, d_model)
      -> Linear(d_model -> pred_len*D) -> reshape
      -> denorm
    (B, pred_len, D)

References:
    SeqSNN (Microsoft, MIT License): https://github.com/microsoft/SeqSNN
      SeqSNN-RPE/network/snn/spikformer.py
      SeqSNN-RPE/network/snn/spikformer_CPG.py
      SeqSNN-RPE/network/snn/spikformer_xnor.py
      SeqSNN-RPE/network/snn/spikformer_xnor_gray.py
      SeqSNN-RPE/network/snn/spikformer_xnor_log.py
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from models.layers.spike_attention import Block, SSA_XNOR, MLPCPG
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding, CPGLinear

_XNOR_TYPES = ('xnor', 'xnor_gray', 'xnor_log')
_ATTN_PE_MAP = {'xnor': 'none', 'xnor_gray': 'gray', 'xnor_log': 'log'}


class Spikformer(nn.Module):
    """Unified spiking transformer for long-term time-series forecasting.

    Args:
        input_len:       Sequence length (L = seq_len).
        T:               Number of SNN time steps.
        blocks:          Number of transformer blocks.
        D:               Number of input/output channels (enc_in).
        pred_len:        Prediction horizon.
        tau:             LIF membrane time constant.
        d_model:         Attention / embedding dimension (must be divisible by heads).
        d_ff:            Feedforward hidden dim (default: 4 x d_model).
        heads:           Number of attention heads (default 8).
        common_thr:      LIF firing threshold (default 1.0).
        qk_scale:        Q·K scale for standard SSA (default 0.125); ignored for XNOR.
        encoder_type:    Spike encoder — 'conv' or 'delta'.
        pe_type:         Positional encoding — 'none', 'learn', 'static', 'conv',
                         'neuron', 'random'. XNOR variants require 'conv'; CPG requires 'none'.
        pe_mode:         How PE is combined — 'add' (default) or 'concat'.
                         'concat' widens the input projection for pe_type in {'neuron','random'}.
                         Only meaningful for 'standard' variant (XNOR forces 'add'; CPG ignores).
        attn_type:       Attention variant — 'standard', 'cpg', 'xnor', 'xnor_gray', 'xnor_log'.
        gray_bits:       Gray-code bits appended to Q/K for 'xnor_gray' (default 10).
        num_pe_neuron:   CPG oscillator neurons / PE neurons for neuron/random PE (default 10).
        neuron_pe_scale: Frequency scale for CPG / neuron PE (default 1000.0).
        dropout:         Dropout on the global embedding before out_proj (default 0.1).
        normalize:       RevIN-style per-instance normalization (default True).
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
        pe_type: str = 'none',
        pe_mode: str = 'add',
        attn_type: str = 'standard',
        gray_bits: int = 10,
        num_pe_neuron: int = 10,
        neuron_pe_scale: float = 1000.0,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        assert attn_type in ('standard', 'cpg', *_XNOR_TYPES), (
            f"attn_type must be 'standard', 'cpg', 'xnor', 'xnor_gray', or 'xnor_log', "
            f"got {attn_type!r}"
        )
        if attn_type in _XNOR_TYPES:
            assert pe_type == 'conv', (
                f"XNOR attention variants require pe_type='conv', got {pe_type!r}"
            )
        if attn_type == 'cpg':
            assert pe_type == 'none', (
                "CPG uses built-in positional encoding via CPGLinear; set pe_type='none'"
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

        # Positional encoding
        # XNOR variants: conv PE (required)
        # standard: any pe_type; pe_mode='concat' widens D for neuron/random
        # cpg: no PE 
        if pe_type != 'none':
            _pe_mode = 'add' if attn_type in _XNOR_TYPES else pe_mode
            self.pe = PositionEmbedding(
                input_size=D,
                pe_type=pe_type,
                pe_mode=_pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                num_steps=T,
            )

        # Input embedding dimension (widened when concat PE is active)
        _concat_pe = (
            attn_type == 'standard'
            and pe_type in ('neuron', 'random')
            and pe_mode == 'concat'
        )
        embed_in = D + num_pe_neuron if _concat_pe else D

        # Input embedding + init spike
        if attn_type == 'cpg':
            # CPGLinear fuses sinusoidal position bias; no separate PE, no init_bn
            self.input_embed = CPGLinear(
                input_size=D,
                output_size=d_model,
                num_pe_neuron=num_pe_neuron,
                w_max=neuron_pe_scale,
                dropout=dropout,
            )
        else:
            self.input_embed = nn.Linear(embed_in, d_model)
            # init_bn only for XNOR variants (standard skips, matching spikformer.py)
            if attn_type in _XNOR_TYPES:
                self.init_bn = nn.BatchNorm1d(d_model)

        self.init_lif = neuron.LIFNode(
            tau=tau, step_mode='m', detach_reset=True,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr, backend='torch',
        )

        # Transformer blocks
        _mlp_cls = MLPCPG if attn_type == 'cpg' else None
        _mlp_kwargs = (
            {'num_pe_neuron': num_pe_neuron, 'neuron_pe_scale': neuron_pe_scale}
            if attn_type == 'cpg' else {}
        )
        if attn_type in _XNOR_TYPES:
            self.transformer_blocks = nn.ModuleList([
                Block(length=input_len, tau=tau, common_thr=common_thr,
                      dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale,
                      attn_cls=SSA_XNOR,
                      attn_pe=_ATTN_PE_MAP[attn_type],
                      gray_bits=gray_bits)
                for _ in range(blocks)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                Block(length=input_len, tau=tau, common_thr=common_thr,
                      dim=d_model, d_ff=d_ff, heads=heads, qk_scale=qk_scale,
                      mlp_cls=_mlp_cls, mlp_kwargs=_mlp_kwargs)
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

        # Spike encoding: (B, L, D) -> (T, B, D, L) -> (T, B, L, D)
        x = self.spike_encoder(x)
        x = x.transpose(-2, -1)                                        # (T, B, L, D)
        T, _, L, _ = x.shape

        # Optional positional encoding
        if self.pe_type != 'none':
            x = self.pe(x)                                             # (T, B, L, D')

        # Input embedding + [BN for XNOR] + LIF
        if self.attn_type == 'cpg':
            # CPGLinear expects (B, TL, D); encodes T*L positions via CPG bias
            x = x.permute(1, 0, 2, 3).flatten(1, 2)                   # (B, TL, D)
            x = self.input_embed(x)                                    # (B, TL, d_model)
            x = x.reshape(B, T, L, -1).permute(1, 0, 2, 3)            # (T, B, L, d_model)
        else:
            x = self.input_embed(x.flatten(0, 1))                      # (TB, L, d_model)
            if self.attn_type in _XNOR_TYPES:
                x = self.init_bn(x.transpose(-1, -2)).transpose(-1, -2)
            x = x.reshape(T, B, L, -1).contiguous()                   # (T, B, L, d_model)

        x = self.init_lif(x)                                           # (T, B, L, d_model)

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
