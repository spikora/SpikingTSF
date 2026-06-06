"""TSLIFNode — two-compartment time-series LIF neuron.

Reference:
    arXiv:2503.05108 (TS-LIF paper)
    TS-LIF/TS-LIF/SeqSNN/network/snn/TSLIF.py (TSLIFNode class)

Innovation over standard LIF:
  - Two compartments: v1 (dendritic/'long') and v2 (somatic/'short')
  - Dual-compartment charge dynamics with learnable decay and cross-coupling:
        v1 ← d0·v1 + d1·x − yy·v2   (dendritic)
        v2 ← d2·v2 + d3·x − kk·v1   (somatic)
  - Dual spike output: s_s = ATan(v2−θ), s_l = ATan(v1−θ)
  - Learnable scalar combination: spike = α_s·s_s + α_l·s_l
  - Soft reset: v1 −= γ·s_l,  v2 −= θ·s_s

Design choices vs. reference:
  - alpha_s/alpha_l are scalar (not per-dim [1,D]) so the node works for any
    input tensor shape — the reference had hardcoded dims (128 or 168) that are
    dataset-specific and do not generalise.
  - State v1/v2 is lazily initialised on first call and reset via reset(),
    which is compatible with spikingjelly functional.reset_net().
  - Uses spikingjelly surrogate.ATan() (same gradient as snntorch atan).

Usage:
    node = TSLIFNode(v_threshold=1.0, gamma=0.5)
    # Works on any shaped tensor; state accumulates within one forward() call
    out = node(x)   # x: (T, B, L, D) or (N, C, L) or any shape
    # Call functional.reset_net(model) at the start of model.forward() to reset.
"""

import torch
from torch import nn
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.base import MemoryModule


class TSLIFNode(MemoryModule):
    """Two-compartment TS-LIF neuron.

    Applies the dual-compartment dynamics to the entire input tensor in one
    call (treating all dimensions as batch). State v1/v2 persists between
    calls within the same forward pass and is reset via reset() or by
    spikingjelly's functional.reset_net(). Extends spikingjelly MemoryModule
    so functional.reset_net() recognises it without warnings.

    Args:
        v_threshold: Firing threshold θ for both compartments (default 1.0).
        gamma:       Soft-reset coefficient for dendritic compartment (default 0.5).
    """

    def __init__(self, v_threshold: float = 1.0, gamma: float = 0.5):
        super().__init__()
        self.v_threshold = v_threshold
        self.gamma = gamma
        self.surrogate_fn = surrogate.ATan()

        # Learnable decay factors: [d0, d1, d2, d3]
        #   v1 ← d0·v1 + d1·x − yy·v2
        #   v2 ← d2·v2 + d3·x − kk·v1
        self.decay_factor = nn.Parameter(torch.tensor([0.8, 0.2, 0.3, 0.7]))
        self.kk = nn.Parameter(torch.tensor([0.8]))   # soma→dendrite coupling
        self.yy = nn.Parameter(torch.tensor([0.1]))   # dendrite→soma coupling

        # Learnable scalar spike-mix weights (scalar for shape generality)
        self.alpha_s = nn.Parameter(torch.ones(1))    # weight on somatic spike
        self.alpha_l = nn.Parameter(torch.ones(1))    # weight on dendritic spike

        # Membrane state registered via MemoryModule for proper reset tracking
        self.register_memory('v1', 0.)
        self.register_memory('v2', 0.)

    def _init_state(self, x: torch.Tensor):
        if isinstance(self.v1, float):
            self.v1 = torch.zeros_like(x)
            self.v2 = torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._init_state(x)

        # Two-compartment charge update
        v1_new = (self.decay_factor[0] * self.v1
                  + self.decay_factor[1] * x
                  - self.yy * self.v2)
        v2_new = (self.decay_factor[2] * self.v2
                  + self.decay_factor[3] * x
                  - self.kk * self.v1)
        self.v1 = v1_new
        self.v2 = v2_new

        # Dual fire via surrogate gradient
        s_s = self.surrogate_fn(self.v2 - self.v_threshold)   # somatic spike
        s_l = self.surrogate_fn(self.v1 - self.v_threshold)   # dendritic spike

        # Learnable scalar combination
        spike = self.alpha_s * s_s + self.alpha_l * s_l

        # Soft reset
        self.v1 = self.v1 - self.gamma * s_l
        self.v2 = self.v2 - self.v_threshold * s_s

        return spike
