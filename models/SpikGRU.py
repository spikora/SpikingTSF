"""SpikeGRU model implementation.

Reference:
    Lv, Changze, Yansen Wang, Dongqi Han, Xiaoqing Zheng,
    Xuanjing Huang, and Dongsheng Li.
    "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks."
    arXiv preprint arXiv:2402.01533, 2024.
    Paper: https://arxiv.org/abs/2402.01533

Official project repository (SeqSNN):
    https://github.com/microsoft/SeqSNN  (SeqSNN/network/snn/spikegru.py)

Architecture:
  1. Spike encoder (conv or delta)            -> (T, B, D, L)
  2. Transpose                                -> (T, B, L, D)
  3. Optional positional encoding
  4. Linear input projection + init LIF       -> (T, B, L, hidden_dim)
  5. Stack of SpikeGRUCell                    -> (T, B, L, hidden_dim)
     GRU gates (r, z, n) via ATan surrogate; hidden state spike-quantized by LIF
  6. Decoder: dense1 -> BN -> LIF -> dense2      -> (T, B, pred_len, D)
  7. Mean T                                   -> (B, pred_len, D)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding


def _make_lif(tau: float) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=tau, step_mode='m', detach_reset=True,
        surrogate_function=surrogate.ATan(),
    )


class SpikeGRUCell(nn.Module):
    """GRU spiking cell with input-to-hidden and hidden-to-hidden paths.
    """

    def __init__(self, input_size: int, hidden_size: int, tau: float):
        super().__init__()
        self.hidden_size = hidden_size
        self._atan = surrogate.ATan()
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        # Single-step LIF: state is h itself (explicitly maintained below)
        self.lif = neuron.LIFNode(
            tau=tau, step_mode='s', detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, L, input_size)
        T, B, L, _ = x.shape
        BL = B * L
        h = x.new_zeros(BL, self.hidden_size)                   # initial hidden state

        outputs = []
        for t in range(T):
            xf = x[t].reshape(BL, -1)                          # (BL, input_size)

            y_ih = self.linear_ih(xf).chunk(3, dim=-1)         # 3 x (BL, hidden_size)
            y_hh = self.linear_hh(h).chunk(3, dim=-1)

            r = self._atan(y_ih[0] + y_hh[0])                  # reset gate 
            z = self._atan(y_ih[1] + y_hh[1])                  # update gate 
            n = self._atan(y_ih[2] + r * y_hh[2])              # candidate

            h_new = (1.0 - z) * n + z * h                      # gated update (BL, H)
            h = self.lif(h_new.reshape(B, L, self.hidden_size)) # (B, L, H) — LIF spike
            h = h.reshape(BL, self.hidden_size)
            outputs.append(h.reshape(B, L, self.hidden_size))

        return torch.stack(outputs, dim=0)                      # (T, B, L, hidden_size)


class SpikeGRU(nn.Module):
    """SpikeGRU: gated spiking RNN for long-term time-series forecasting.

    Args:
        input_len:      Sequence length (L = seq_len).
        T:              Number of SNN time steps.
        blocks:         Number of SpikeGRUCell layers.
        D:              Number of input/output channels (enc_in).
        pred_len:       Prediction horizon.
        tau:            LIF membrane time constant.
        hidden_dim:     Recurrent cell and decoder hidden size
                        (mapped from alpha in args).
        encoder_type:   Spike encoder — 'conv' or 'delta'.
        pe_type:        Positional encoding type — 'none', 'learn',
                        'static', 'conv', 'neuron', 'random'.
        pe_mode:        How PE is combined — 'add' or 'concat'.
                        'concat' widens the input projection for
                        'neuron'/'random' PE types.
        num_pe_neuron:  PE neurons for 'neuron'/'random' PE in concat mode.
        neuron_pe_scale: Frequency scale for neuron PE.
        normalize:      RevIN-style per-instance normalization.
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

        # Spike encoder: (B, L, D) -> (T, B, D, L)
        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        # Positional encoding
        if pe_type != 'none':
            self.pe = PositionEmbedding(
                input_size=D,
                pe_type=pe_type,
                pe_mode=pe_mode,
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                num_steps=T,
            )

        # Input projection: D (possibly widened by concat PE) -> hidden_dim
        proj_in = (
            D + num_pe_neuron
            if (pe_type in ('neuron', 'random') and pe_mode == 'concat')
            else D
        )
        self.input_proj = nn.Linear(proj_in, hidden_dim)
        self.init_lif = _make_lif(tau)

        # Stacked GRU cells
        self.net = nn.ModuleList([
            SpikeGRUCell(hidden_dim, hidden_dim, tau) for _ in range(blocks)
        ])

        # Decoder: hidden_dim -> D -> BN -> LIF -> L -> pred_len
        self.dense1 = nn.Linear(hidden_dim, D)
        self.bn = nn.BatchNorm1d(D)
        self.out_lif = _make_lif(tau)
        self.dense2 = nn.Linear(input_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()            # (B, 1, D)
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()                                              # (B, 1, D)
            x = x / std

        # Spike encoding
        h = self.spike_encoder(x)                                  # (T, B, D, L)
        h = h.transpose(-1, -2)                                    # (T, B, L, D)
        T, B, L, _ = h.shape

        # Positional encoding
        if self.pe_type != 'none':
            h = self.pe(h)                                         # (T, B, L, D')

        # Input projection + init spike
        h = self.input_proj(h.flatten(0, 1)).reshape(T, B, L, -1) # (T, B, L, hidden_dim)
        h = self.init_lif(h)

        # Stacked GRU cells
        for cell in self.net:
            h = cell(h)                                            # (T, B, L, hidden_dim)

        # Decoder: hidden_dim -> D -> BN -> LIF -> L -> pred_len
        h = self.dense1(h)                                         # (T, B, L, D)
        h = h.flatten(0, 1).permute(0, 2, 1)                      # (TB, D, L)
        h = self.bn(h)
        h = h.permute(0, 2, 1).reshape(T, B, L, -1)               # (T, B, L, D)
        h = self.out_lif(h)

        h = h.permute(0, 1, 3, 2)                                 # (T, B, D, L)
        h = self.dense2(h)                                         # (T, B, D, pred_len)
        h = h.permute(0, 1, 3, 2)                                 # (T, B, pred_len, D)

        out = h.mean(dim=0)                                        # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean

        return out
