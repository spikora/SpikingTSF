"""TSGRU — TS-LIF Spiking GRU for long-term time-series forecasting.

Reference:
    arXiv:2503.05108 (TS-LIF paper)
    TS-LIF/TS-LIF/SeqSNN/network/snn/spikegru.py (ITSSNNGRU2D class)

Adaptation of SpikeGRU (models/SpikGRU.py) that replaces every spikingjelly
LIFNode with TSLIFNode — the two-compartment dual-spike neuron introduced by
the TS-LIF paper. All other architectural choices (encoder, PE, GRU gating,
decoder) follow our existing SpikeGRU adaptation of SeqSNN.

Key differences from SpikeGRU:
  - init_lif        → TSLIFNode()   (was step_mode='m' LIFNode)
  - SpikeGRUCell.lif → TSLIFNode()  (was step_mode='s' LIFNode)
  - out_lif         → TSLIFNode()   (was step_mode='m' LIFNode)
  - functional.reset_net(self) called at the start of forward() to reset
    TSLIFNode membrane state (v1, v2) between batches.

Architecture (same pipeline as SpikeGRU):
    (B, L, D)
      → RevIN norm
      → SpikeEncoder           (T, B, D, L)
      → transpose              (T, B, L, D)
      → [optional PE]
      → Linear(D→hidden) + TSLIFNode   (T, B, L, hidden)
      → TSGRUCell × blocks     (T, B, L, hidden)   sequential T-step loop
          GRU gates (r,z,n) via ATan; hidden TSLIFNode-quantized
      → Linear(hidden→D) + BN + TSLIFNode  (T, B, L, D)
      → Linear(L→pred_len)     (T, B, D, pred_len)
      → permute → mean T       (B, pred_len, D)
      → denorm
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, functional

from models.layers.tslif import TSLIFNode
from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding


class TSGRUCell(nn.Module):
    """GRU spiking cell with TSLIFNode for hidden-state quantisation.

    Mirrors SpikeGRUCell exactly but replaces the single-step LIFNode with
    TSLIFNode so that the hidden state carries two-compartment temporal memory.
    TSLIFNode state (v1, v2) accumulates across T steps in the sequential loop,
    giving the cell richer temporal dynamics than a memoryless surrogate.

    Args:
        input_size:  Input feature dimension.
        hidden_size: Hidden state dimension.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self._atan = surrogate.ATan()
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.tslif = TSLIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, input_size)
        T, B, L, _ = x.shape
        BL = B * L
        h = x.new_zeros(BL, self.hidden_size)

        outputs = []
        for t in range(T):
            xf = x[t].reshape(BL, -1)                          # (BL, input_size)

            y_ih = self.linear_ih(xf).chunk(3, dim=-1)
            y_hh = self.linear_hh(h).chunk(3, dim=-1)

            r = self._atan(y_ih[0] + y_hh[0])                  # reset gate
            z = self._atan(y_ih[1] + y_hh[1])                  # update gate
            n = self._atan(y_ih[2] + r * y_hh[2])              # candidate

            h_new = (1.0 - z) * n + z * h                      # (BL, H)
            # TSLIFNode applied per-step; state accumulates across loop iterations
            h = self.tslif(h_new.reshape(B, L, self.hidden_size))  # (B, L, H)
            h = h.reshape(BL, self.hidden_size)
            outputs.append(h.reshape(B, L, self.hidden_size))

        return torch.stack(outputs, dim=0)                      # (T, B, L, hidden_size)


class TSGRU(nn.Module):
    """TSGRU: TS-LIF gated spiking RNN for long-term time-series forecasting.

    Args:
        input_len:      Sequence length (L = seq_len).
        T:              Number of SNN time steps.
        blocks:         Number of TSGRUCell layers.
        D:              Number of input/output channels (enc_in).
        pred_len:       Prediction horizon.
        tau:            LIF membrane time constant (used by encoder only).
        hidden_dim:     Recurrent cell and decoder hidden size
                        (mapped from ``alpha`` in args).
        encoder_type:   Spike encoder — ``'conv'`` or ``'delta'``.
        pe_type:        Positional encoding — ``'none'``, ``'learn'``,
                        ``'static'``, ``'conv'``, ``'neuron'``, ``'random'``.
        pe_mode:        How PE is combined — ``'add'`` or ``'concat'``.
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
        encoder_type: str = 'conv',
        pe_type: str = 'none',
        pe_mode: str = 'add',
        num_pe_neuron: int = 10,
        neuron_pe_scale: float = 1000.0,
        normalize: bool = True,
    ):
        super().__init__()
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
                input_size=D, pe_type=pe_type, pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron, neuron_pe_scale=neuron_pe_scale,
                num_steps=T,
            )

        proj_in = (
            D + num_pe_neuron
            if (pe_type in ('neuron', 'random') and pe_mode == 'concat')
            else D
        )
        self.input_proj = nn.Linear(proj_in, hidden_dim)
        self.init_tslif = TSLIFNode()

        self.net = nn.ModuleList([
            TSGRUCell(hidden_dim, hidden_dim) for _ in range(blocks)
        ])

        self.dense1  = nn.Linear(hidden_dim, D)
        self.bn      = nn.BatchNorm1d(D)
        self.out_tslif = TSLIFNode()
        self.dense2  = nn.Linear(input_len, pred_len)

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

        h = self.spike_encoder(x)                                  # (T, B, D, L)
        h = h.transpose(-1, -2)                                    # (T, B, L, D)
        T, B, L, _ = h.shape

        if self.pe_type != 'none':
            h = self.pe(h)

        h = self.input_proj(h.flatten(0, 1)).reshape(T, B, L, -1)  # (T, B, L, hidden)
        h = self.init_tslif(h)

        for cell in self.net:
            h = cell(h)                                            # (T, B, L, hidden)

        h = self.dense1(h)                                         # (T, B, L, D)
        h = h.flatten(0, 1).permute(0, 2, 1)                      # (TB, D, L)
        h = self.bn(h)
        h = h.permute(0, 2, 1).reshape(T, B, L, -1)               # (T, B, L, D)
        h = self.out_tslif(h)

        h = h.permute(0, 1, 3, 2)                                  # (T, B, D, L)
        h = self.dense2(h)                                          # (T, B, D, pred_len)
        h = h.permute(0, 1, 3, 2)                                  # (T, B, pred_len, D)

        out = h.mean(dim=0)                                         # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean

        return out
