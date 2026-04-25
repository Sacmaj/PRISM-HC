"""Frozen cross-modal anchor core.

A `register_buffer` matrix `U` (orthonormal columns, QR-decomposed from a
seeded random matrix) defines a protected k-dimensional subspace inside the
hidden state. Adapter updates are projected away from `span(U)` so that
plasticity cannot drift the anchor coordinates.

`drift` is the per-step subspace residual ||h - U U^T h||, averaged across
layers and batch elements. It does NOT compare to a stored reference
trajectory — it measures how far the current hidden state lies outside the
frozen safety subspace, which the LATCH controller uses as 1 - drift to
estimate instantaneous safety.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .state import SafetyState


class FrozenAnchorCore(nn.Module):
    def __init__(self, d_hidden: int, n_anchors: int, seed: int):
        super().__init__()
        if n_anchors > d_hidden:
            raise ValueError(
                f"n_anchors ({n_anchors}) must be <= d_hidden ({d_hidden})."
            )
        gen = torch.Generator().manual_seed(seed)
        raw = torch.randn(d_hidden, n_anchors, generator=gen)
        q, _ = torch.linalg.qr(raw)
        self.register_buffer("U", q)

    @torch.no_grad()
    def compute_coords(self, h_dict: Dict[int, torch.Tensor]) -> SafetyState:
        coords: Dict[int, torch.Tensor] = {}
        drift_terms = []
        for layer, h in h_dict.items():
            c = h @ self.U
            coords[layer] = c
            recon = c @ self.U.T
            residual = (h - recon).norm(dim=-1)
            drift_terms.append(residual.mean())
        drift = torch.stack(drift_terms).mean()
        canary_margin = float(torch.clamp(1.0 - drift, 0.0, 1.0).item())
        return SafetyState(
            anchor_coords=coords,
            drift=drift,
            canary_margin=canary_margin,
        )

    @torch.no_grad()
    def drift_loss(self, h_new: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
        """L_drift = ||U^T (h_new - h_ref)||^2 — anchor-space drift."""
        delta = (h_new - h_ref) @ self.U
        return delta.pow(2).sum(dim=-1).mean()
