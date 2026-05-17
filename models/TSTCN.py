"""TSTCN — TS-LIF Spiking Temporal Convolutional Network.

Reference:
    arXiv:2503.05108 (TS-LIF paper)
    TS-LIF/TS-LIF/SeqSNN/network/snn/spike_tcn.py (SpikeTemporalConvNet2D class)

Adaptation of SpikeTCN (models/SpikTCN.py) that replaces every spikingjelly
single-step LIFNode with TSLIFNode — the two-compartment dual-spike neuron.
All other architectural choices (Conv1d, channel-independent pipeline, dilated
causal convolutions, SEW residual) follow our existing SpikeTCN adaptation.

Key differences from SpikeTCN:
  - SpikeTCNBlock.lif1/lif2/res_lif → TSLIFNode() (were step_mode='s' LIFNode)
  - functional.reset_net(self) called at start of forward() to reset TSLIFNode
    membrane state (v1, v2) between batches.

Note on TSLIFNode in TCN:
  In SpikeTCN, the single-step LIF was memoryless per call (step_mode='s'),
  acting as a pure surrogate threshold. With TSLIFNode, each block's neuron
  accumulates two-compartment state across the sequential block calls within
  one forward pass. This gives the TCN cross-block temporal memory — closer to
  the TS-LIF paper's intent of richer temporal dynamics at each spiking site.

Architecture (same pipeline as SpikeTCN):
    (B, L, D)
      → RevIN norm
      → SpikeEncoder                    (T, B, D, L)
      → [optional PE]
      → flatten + unsqueeze             (T*B*D, 1, L)
      → Conv1d(1→hidden)                (T*B*D, hidden, L)
      → TSTCNBlock × blocks             (T*B*D, hidden, L)
          causal Conv1d → BN → TSLIFNode, SEW residual with TSLIFNode
      → last causal position            (T*B*D, hidden)
      → Linear(hidden→pred_len)         (T*B*D, pred_len)
      → reshape → mean T                (B, pred_len, D)
      → denorm
"""

import torch
from torch import nn
from torch.nn.utils import weight_norm
from spikingjelly.activation_based import functional

from models.layers.tslif import TSLIFNode
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TSTCNBlock(nn.Module):
    """One dilated causal-conv TCN block with TSLIFNode neurons and SEW residual.

    Operates on (T*B*D, channels, L). TSLIFNode is applied per-block call;
    state (v1, v2) accumulates across sequential block calls within one
    forward pass, then is reset by functional.reset_net() before the next.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Initialise before weight_norm so weight_v/weight_g are set from N(0,0.01).
        # Setting .weight.data after weight_norm is a no-op — weight is a computed hook.
        _c1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                        dilation=dilation, padding=padding)
        _c2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                        dilation=dilation, padding=padding)
        nn.init.normal_(_c1.weight, 0, 0.01)
        nn.init.normal_(_c2.weight, 0, 0.01)
        self.conv1   = weight_norm(_c1)
        self.chomp1  = Chomp1d(padding)
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.tslif1  = TSLIFNode()

        self.conv2   = weight_norm(_c2)
        self.chomp2  = Chomp1d(padding)
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.tslif2  = TSLIFNode()

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        self.res_tslif = TSLIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tslif1(self.bn1(self.chomp1(self.conv1(x))))
        out = self.tslif2(self.bn2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.res_tslif(out + res)                             # SEW residual


class TSTCN(nn.Module):
    """TSTCN: TS-LIF spiking TCN for long-term time-series forecasting.

    Args:
        input_len:      Sequence length (L = seq_len).
        T:              Number of SNN time steps.
        blocks:         Number of TSTCNBlock layers (dilation doubles per block).
        D:              Number of input/output channels (enc_in).
        pred_len:       Prediction horizon.
        tau:            LIF time constant (used by encoder only).
        hidden_dim:     TCN hidden channel size (mapped from ``alpha`` in args).
        kernel_size:    Dilated causal conv kernel size (default 3).
        encoder_type:   Spike encoder — ``'conv'`` or ``'delta'``.
        pe_type:        Positional encoding — ``'none'``, ``'learn'``,
                        ``'static'``, ``'conv'``, ``'neuron'``, ``'random'``.
                        Only ``'add'`` mode is supported (concat changes D).
        num_pe_neuron:  PE neurons for ``neuron``/``random`` PE (default 10).
        neuron_pe_scale: Frequency scale for neuron PE (default 1000.0).
        normalize:      RevIN-style instance normalization (default True).
    """

    def __init__(
        self,
        input_len: int,
        T: int,
        blocks: int,
        D: int,
        pred_len: int,
        tau: float,
        hidden_dim: int,
        kernel_size: int = 3,
        encoder_type: str = 'conv',
        pe_type: str = 'none',
        num_pe_neuron: int = 10,
        neuron_pe_scale: float = 1000.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.T = T
        self.pe_type = pe_type
        self.normalize = normalize

        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        if pe_type != 'none':
            self.pe = PositionEmbedding(
                input_size=D, pe_type=pe_type, pe_mode='add',
                num_pe_neuron=num_pe_neuron, neuron_pe_scale=neuron_pe_scale,
                num_steps=T,
            )

        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)

        self.tcn_blocks = nn.ModuleList([
            TSTCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=2 ** i)
            for i in range(blocks)
        ])

        self.out_proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / std

        B, L, D = x.shape

        functional.reset_net(self)

        h = self.spike_encoder(x)                                   # (T, B, D, L)

        if self.pe_type != 'none':
            h = self.pe(h.transpose(-1, -2)).transpose(-1, -2)     # (T, B, D, L)

        h = h.flatten(0, 2).unsqueeze(1)                           # (T*B*D, 1, L)
        h = self.input_proj(h)                                     # (T*B*D, hidden, L)

        for block in self.tcn_blocks:
            h = block(h)                                           # (T*B*D, hidden, L)

        h = h[:, :, -1]                                            # (T*B*D, hidden)
        h = self.out_proj(h)                                       # (T*B*D, pred_len)

        h = h.reshape(self.T, B, D, -1).permute(0, 1, 3, 2)       # (T, B, pred_len, D)
        out = h.mean(dim=0)                                         # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean

        return out
