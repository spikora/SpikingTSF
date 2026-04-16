"""
Spiking GRU for long-term time series forecasting.

Adapted from SeqSNN (Microsoft, MIT License):
  Lv et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks", ICML 2024.
  https://github.com/microsoft/SeqSNN  (SeqSNN/network/snn/spikegru.py)

Key differences from SeqSNN TSSNNGRU:
  - No runner/registry dependency; returns (B, pred_len, D) directly
  - Uses spikingjelly clock_driven API instead of snntorch (consistent with SpikF)
  - Channel-independent: each of the D variates processed by its own GRU thread
  - GRU gates use standard sigmoid/tanh; the hidden state is quantised by a spiking
    LIF neuron after each GRU step (spike-on-output paradigm)
"""

import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class SpikeGRUCell(nn.Module):
    """
    One-step GRU where the updated hidden state is quantised by a spiking LIF node.

    The GRU equations use standard differentiable gates (sigmoid/tanh).
    The LIF applies spike coding to the output hidden state, providing sparse
    temporal representation and gradient regularisation.
    """

    def __init__(self, input_size: int, hidden_size: int, T: int, tau: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.T = T

        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        # LIF on the new hidden state to produce spike-coded representation
        self.lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (BD, input_size)   h: (BD, hidden_size)
        y_ih = self.linear_ih(x).chunk(3, dim=-1)
        y_hh = self.linear_hh(h).chunk(3, dim=-1)

        r = torch.sigmoid(y_ih[0] + y_hh[0])          # reset gate
        z = torch.sigmoid(y_ih[1] + y_hh[1])          # update gate
        n = torch.tanh(y_ih[2] + r * y_hh[2])         # candidate hidden
        h_new = (1.0 - z) * n + z * h                 # updated hidden

        # Spike quantisation over T time steps (static input repeated T times)
        h_t = h_new.unsqueeze(0).repeat(self.T, 1, 1)  # (T, BD, hidden)
        spk = self.lif(h_t)                             # (T, BD, hidden)
        return spk.mean(0)                              # (BD, hidden)


class Model(nn.Module):
    """
    SpikGRU: channel-independent Spiking GRU.

    Each variate is processed independently along the sequence dimension.
    The hidden state after T_seq GRU steps is projected to pred_len.

    Args (from configs namespace):
        seq_len    : input length L
        pred_len   : forecast horizon H
        enc_in     : number of input variables D
        hidden_dim : GRU hidden size
        levels     : number of stacked GRU layers
        T          : SNN time-steps for the LIF quantisation
        tau        : LIF membrane time constant
        patch_dim  : input embedding size (1 → patch_dim via Conv1d)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        D = configs.enc_in
        hidden = configs.hidden_dim
        patch_dim = configs.patch_dim
        T = configs.T
        tau = configs.tau
        layers = configs.levels
        dropout = getattr(configs, 'dropout', 0.1)

        # Embed each variate: (BD, 1, L) → (BD, patch_dim, L) → (BD, L, patch_dim)
        self.input_proj = nn.Conv1d(1, patch_dim, kernel_size=1)

        # Stacked GRU cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            in_size = patch_dim if i == 0 else hidden
            self.cells.append(SpikeGRUCell(in_size, hidden, T, tau))

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(D)
        self.out_proj = nn.Linear(hidden, configs.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        B, L, D = x.shape

        # Channel-independent: (B, L, D) → (BD, 1, L) → embed → (BD, L, patch_dim)
        x = x.permute(0, 2, 1).reshape(B * D, 1, L)
        x = self.input_proj(x).permute(0, 2, 1).contiguous()  # (BD, L, patch_dim)

        # Unroll GRU over sequence
        h_list = [torch.zeros(B * D, cell.hidden_size, device=x.device)
                  for cell in self.cells]

        for t in range(L):
            inp = x[:, t, :]  # (BD, patch_dim or hidden)
            for i, cell in enumerate(self.cells):
                h_list[i] = cell(inp, h_list[i])
                inp = self.dropout(h_list[i])

        # Final hidden state → forecast
        h_last = h_list[-1]                                         # (BD, hidden)
        out = self.out_proj(h_last)                                 # (BD, pred_len)
        out = out.reshape(B, D, self.pred_len).permute(0, 2, 1)    # (B, pred_len, D)

        return out * std + mean
