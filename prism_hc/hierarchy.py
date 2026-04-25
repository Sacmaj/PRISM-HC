"""Toy hierarchical predictive-processing stack.

Belief structure:
  - bottom_up(x):  x -> h_0 -> h_1 -> ... -> h_{L-1}     (per-layer linear+tanh)
  - predict(l, mu): identity                              (prediction = belief)
  - errors:        eps_l = h_l - mu_l
  - free_energy:   sum_l 0.5 (Pi_l * eps_l^2).sum() - 0.5 log(Pi_l).sum()  (diag Pi)
  - readout(mu_0): linear projection back to d_in for the demo's "y" output.

Identity prediction keeps the prototype minimal — the hierarchical dynamics
are still exercised through the PGSTR gate's prior/evidence mixing.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import PrismConfig


class BeliefHierarchy(nn.Module):
    def __init__(self, cfg: PrismConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        prev = cfg.d_in
        for _ in range(cfg.L):
            layers.append(nn.Linear(prev, cfg.d_hidden, bias=False))
            prev = cfg.d_hidden
        self.bottom_up_layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self.output = nn.Linear(cfg.d_hidden, cfg.d_in, bias=False)

    def bottom_up(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        h_dict: Dict[int, torch.Tensor] = {}
        h = x
        for l, layer in enumerate(self.bottom_up_layers):
            h = self.activation(layer(h))
            h_dict[l] = h
        return h_dict

    def predict(self, layer: int, mu: torch.Tensor) -> torch.Tensor:
        # Identity generative model: prediction equals belief.
        return mu

    def free_energy(
        self,
        pi_l_dict: Dict[int, torch.Tensor],
        eps_l_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        # Pin the accumulator to the same device/dtype as model parameters so
        # CUDA / MPS runs do not hit a device-mismatch error on F + quad.
        ref = next(self.parameters())
        F = torch.zeros((), dtype=ref.dtype, device=ref.device)
        for l, eps in eps_l_dict.items():
            pi = pi_l_dict[l]
            quad = 0.5 * (pi * eps.pow(2)).sum(dim=-1).mean()
            logdet = -0.5 * torch.log(pi).sum(dim=-1).mean()
            F = F + quad + logdet
        return F

    def readout(self, mu_top: torch.Tensor) -> torch.Tensor:
        return self.output(mu_top)
