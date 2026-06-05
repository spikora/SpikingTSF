"""SpikeTCN model implementation.

Reference:
    Lv, Changze, Yansen Wang, Dongqi Han, Xiaoqing Zheng,
    Xuanjing Huang, and Dongsheng Li.
    "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks."
    arXiv preprint arXiv:2402.01533, 2024.
    Paper: https://arxiv.org/abs/2402.01533

Official project repository (SeqSNN):
    https://github.com/microsoft/SeqSNN  (SeqSNN/network/snn/spike_tcn.py)

Architecture follows SeqSNN's SpikeTemporalConvNet2D, adapted to spikingjelly
activation_based API and our codebase conventions:
  1. Spike encoder (conv or delta)         → (T, B, D, L)
  2. Optional positional encoding          → (T, B, D, L)
  3. Channel-independent input projection  → (T*B*D, hidden_dim, L)
  4. Stack of SpikeTCNBlock                → (T*B*D, hidden_dim, L)
     dilated causal Conv1d → BN → single-step LIF, with SEW residual
  5. Last causal position → out_proj       → (T*B*D, pred_len)
  6. Reshape → mean T                      → (B, pred_len, D)
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

from models.layers.spike_encoder import ConvEncoder, DeltaEncoder
from models.layers.positional_encoding import PositionEmbedding


def _make_lif_m(tau: float) -> neuron.LIFNode:
    """Multi-step LIF — state persists across T steps (used in encoder)."""
    return neuron.LIFNode(
        tau=tau, step_mode='m', detach_reset=True,
        surrogate_function=surrogate.ATan(),
    )


def _make_lif_s(tau: float) -> neuron.LIFNode:
    """Single-step LIF — stateless per call; reset each forward (used in TCN blocks)."""
    return neuron.LIFNode(
        tau=tau, step_mode='s', detach_reset=True,
        surrogate_function=surrogate.ATan(),
    )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class SpikeTCNBlock(nn.Module):
    """One dilated causal-conv TCN block with two spiking LIF layers and SEW residual.

    Operates on (T*B*D, channels, L) where T is folded into the batch dimension.
    Single-step LIF is applied directly to the (T*B*D, C, L) tensor — each sample
    in the batch (including each T replicate) is treated independently, so membrane
    state does NOT persist across SNN time steps. This matches the paper's rule that
    TCN membrane potentials reset at each time-series step, enabling parallel training.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, tau: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.lif1 = _make_lif_s(tau)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lif2 = _make_lif_s(tau)

        # SEW down-sample if channel sizes differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
        self.res_lif = _make_lif_s(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T*B*D, in_channels, L) — LIF applied per-sample, no T-unfolding
        out = self.lif1(self.bn1(self.chomp1(self.conv1(x))))
        out = self.lif2(self.bn2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.res_lif(out + res)                       # SEW residual


class SpikeTCN(nn.Module):
    """SpikeTCN: spiking temporal convolutional network for long-term forecasting.

    Args:
        input_len:      Sequence length (L = seq_len).
        T:              Number of SNN time steps.
        blocks:         Number of TCN blocks (dilations: 2^0, 2^1, …, 2^(blocks-1)).
        D:              Number of input/output channels (enc_in).
        pred_len:       Prediction horizon.
        tau:            LIF membrane time constant.
        hidden_dim:     TCN hidden channel size (mapped from ``alpha`` in args).
        kernel_size:    Dilated causal conv kernel size (default 3).
        encoder_type:   Spike encoder — ``'conv'`` or ``'delta'``.
        pe_type:        Positional encoding type — ``'none'``, ``'learn'``,
                        ``'static'``, ``'conv'``, ``'neuron'``, ``'random'``.
                        Note: only ``pe_mode='add'`` is supported (``'concat'``
                        would change the channel-independent D dimension).
        num_pe_neuron:  PE neurons for ``neuron``/``random`` PE in add mode.
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

        # Spike encoder: (B, L, D) → (T, B, D, L)
        if encoder_type == 'conv':
            self.spike_encoder = ConvEncoder(output_size=T, tau=tau)
        elif encoder_type == 'delta':
            self.spike_encoder = DeltaEncoder(output_size=T, tau=tau)
        else:
            raise ValueError(f'Unknown encoder_type: {encoder_type!r}')

        # Positional encoding (add mode only — concat would change D)
        if pe_type != 'none':
            self.pe = PositionEmbedding(
                input_size=D,
                pe_type=pe_type,
                pe_mode='add',
                num_pe_neuron=num_pe_neuron,
                neuron_pe_scale=neuron_pe_scale,
                num_steps=T,
            )

        # Channel-independent input projection: 1 → hidden_dim
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)

        # Stacked TCN blocks with exponentially growing dilation
        self.tcn_blocks = nn.ModuleList([
            SpikeTCNBlock(hidden_dim, hidden_dim, kernel_size,
                          dilation=2 ** i, tau=tau)
            for i in range(blocks)
        ])

        # Decoder: last causal position → pred_len, then mean over T
        self.out_proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True).detach()            # (B, 1, D)
            x = x - mean
            std = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()                                              # (B, 1, D)
            x = x / std

        B, L, D = x.shape

        # Spike encoding: (B, L, D) → (T, B, D, L)
        h = self.spike_encoder(x)

        # Optional PE: (T, B, D, L) → transpose → (T, B, L, D) → PE → transpose back
        if self.pe_type != 'none':
            h = self.pe(h.transpose(-1, -2)).transpose(-1, -2)    # (T, B, D, L)

        # Channel-independent TCN:
        # (T, B, D, L) → (T*B*D, 1, L) → input_proj → (T*B*D, hidden_dim, L)
        h = h.flatten(0, 2).unsqueeze(1)                          # (T*B*D, 1, L)
        h = self.input_proj(h)                                    # (T*B*D, hidden_dim, L)

        for block in self.tcn_blocks:
            h = block(h)                                          # (T*B*D, hidden_dim, L)

        # Causal output at last position captures full history
        h = h[:, :, -1]                                           # (T*B*D, hidden_dim)
        h = self.out_proj(h)                                      # (T*B*D, pred_len)

        h = h.reshape(self.T, B, D, -1).permute(0, 1, 3, 2)      # (T, B, pred_len, D)
        out = h.mean(dim=0)                                       # (B, pred_len, D)

        if self.normalize:
            out = out * std + mean

        return out
