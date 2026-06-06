"""TSFormer — TS-LIF Inverted Spiking Transformer for long-term forecasting.

Reference:
    arXiv:2503.05108 (TS-LIF paper)
    TS-LIF/TS-LIF/SeqSNN/network/snn/ispikformer.py (iSpikformer class)

Adaptation of iSpikformer (models/iSpikformer.py) that replaces every
spikingjelly LIFNode with TSLIFNode — the two-compartment dual-spike neuron
introduced by the TS-LIF paper. Uses TSBlock (TSSSA + TSMLP) from
models/layers/spike_attention.py, and TSLIFNode inside DataEmbeddingInverted.

All other architectural choices (inverted iTransformer paradigm, channel tokens,
encoder, decoder via h[-1]) follow our iSpikformer adaptation of SeqSNN.

Key differences from iSpikformer:
  - DataEmbeddingInverted.lif → TSLIFNode()   (was step_mode='m' LIFNode)
  - transformer Block         → TSBlock        (TSSSA + TSMLP with TSLIFNode)
  - functional.reset_net(self) called at start of forward()

Architecture (same as iSpikformer):
    (B, L, D)
      → RevIN norm
      → SpikeEncoder             (T, B, D, L)
      → transpose                (T, B, L, D)
      → DataEmbeddingInverted    (T, B, D, d_model)   Linear(L→d_model)+BN+TSLIFNode
      → TSBlock × blocks         (T, B, D, d_model)   TSSSA+TSMLP over D channel tokens
      → last T step              (B, D, d_model)
      → Linear(d_model→pred_len) (B, D, pred_len)
      → transpose                (B, pred_len, D)
      → denorm
"""

import torch
from torch import nn
from spikingjelly.activation_based import functional

from models.layers.tslif import TSLIFNode
from models.layers.spike_attention import TSBlock
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder


class TSDataEmbeddingInverted(nn.Module):
    """Inverted channel-wise temporal embedding with TSLIFNode.

    Same as DataEmbeddingInverted in iSpikformer but uses TSLIFNode instead
    of spikingjelly LIFNode for the post-BN spike gate.

    Input : (T, B, L, D)
    Output: (T, B, D, d_model)
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.bn    = nn.BatchNorm1d(d_model)
        self.tslif = TSLIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, D)
        T, B, L, D = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)                    # (TB, D, L)
        x = self.value_embedding(x)                                  # (TB, D, d_model)
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)          # (TB, D, d_model)
        x = x.reshape(T, B, D, self.d_model)
        return self.tslif(x)                                         # (T, B, D, d_model)


class TSFormer(nn.Module):
    """TSFormer: TS-LIF inverted spiking transformer for long-term forecasting.

    Args:
        input_len:    Sequence length (L = seq_len).
        T:            Number of SNN time steps.
        blocks:       Number of TSBlock transformer layers.
        D:            Number of input/output channels (enc_in).
        pred_len:     Prediction horizon.
        tau:          LIF time constant (used by encoder only).
        d_model:      Per-channel embedding dimension (mapped from ``alpha`` in args).
        d_ff:         Feedforward hidden size in TSMLP (default: d_model × 4).
        heads:        Attention heads in TSSSA (must divide d_model).
        common_thr:   Threshold passed through to TSBlock signatures (not used by
                      TSLIFNode which has its own v_threshold).
        qk_scale:     Q·K scaling factor in TSSSA (default 0.125).
        encoder_type: Spike encoder — ``'conv'`` or ``'delta'``.
        normalize:    RevIN-style instance normalization (default True).
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

        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        self.emb = TSDataEmbeddingInverted(input_len, d_model)

        # TSBlock: TSSSA + TSMLP; attention runs over D channel tokens
        self.attn_blocks = nn.ModuleList([
            TSBlock(
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
            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / std

        functional.reset_net(self)

        h = self.spike_encoder(x)                                   # (T, B, D, L)
        h = h.transpose(-1, -2)                                     # (T, B, L, D)

        h = self.emb(h)                                             # (T, B, D, d_model)

        for blk in self.attn_blocks:
            h = blk(h)                                              # (T, B, D, d_model)

        out = h[-1, :, :, :]                                        # (B, D, d_model)
        out = self.dense(out)                                        # (B, D, pred_len)
        out = out.transpose(-1, -2)                                  # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean

        return out
