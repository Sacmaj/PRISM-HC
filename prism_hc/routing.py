"""PGSTR routing gate.

Mixes top-down prior with bottom-up evidence + reservoir-routed features,
with mixing weight controlled by per-layer plasticity scalar R_l:

    g     = sigmoid(2 * R_l - 1)
    out   = (1 - g) * prior + g * (evidence + routed)

When R_l ~ 0 the layer trusts its prior; when R_l ~ 1 it absorbs evidence.
"""

from __future__ import annotations

import torch


def pgstr_gate(
    prior: torch.Tensor,
    evidence: torch.Tensor,
    routed: torch.Tensor,
    R_l: torch.Tensor,
) -> torch.Tensor:
    g = torch.sigmoid(2.0 * R_l - 1.0)
    return (1.0 - g) * prior + g * (evidence + routed)
