"""Seeded bilinear reservoir + linear readout.

phi(x) = einsum('bi,bj,ijo->bo', x, x, W_rand)        # frozen, gradient-free
out(x) = readout(phi(x))                              # only trainable weight

W_rand is a (d_in, d_in, d_out) tensor sampled once with a fixed seed and
registered as a buffer so it never receives gradients. The readout is a
plain Linear so adapter commits flow into it after null-space projection.
"""

from __future__ import annotations

import torch
from torch import nn


class SeededBilinearReservoir(nn.Module):
    def __init__(self, d_in: int, d_reservoir: int, d_hidden: int, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        scale = 1.0 / max(d_in, 1) ** 0.5
        W = torch.randn(d_in, d_in, d_reservoir, generator=gen) * scale
        self.register_buffer("W_rand", W)
        self.readout = nn.Linear(d_reservoir, d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.einsum("bi,bj,ijo->bo", x, x, self.W_rand)
        return self.readout(phi)
