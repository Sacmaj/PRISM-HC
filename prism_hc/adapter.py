"""Null-space projection adapter.

For a frozen anchor basis U with orthonormal columns:

    g_safe = g - U @ (U.T @ g)

removes any component of the gradient that lies in span(U), so weight
commits cannot drift the protected anchor coordinates.

Authority asymmetry: this module mutates trainable params, never U.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from .state import ControllerState


class NullSpaceAdapter:
    @staticmethod
    @torch.no_grad()
    def project_update(g: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Remove the component of `g` that lies in `span(U)`.

        Convention: `g.shape[-1] == U.shape[0] == d_hidden` for hidden-aligned
        params. For params whose last dim is not d_hidden, we skip projection
        (return `g` unchanged) so that biases / readout weights still update.
        """
        if g.dim() == 0 or g.shape[-1] != U.shape[0]:
            return g
        shape = g.shape
        flat = g.reshape(-1, shape[-1])
        proj = flat - (flat @ U) @ U.T
        return proj.reshape(shape)

    @staticmethod
    @torch.no_grad()
    def apply_commit(
        state: "ControllerState",
        named_params: Dict[str, torch.nn.Parameter],
        grads_safe: Dict[str, torch.Tensor],
        eta_w: float,
        commit_cost: float,
    ) -> None:
        """SGD commit scaled by eta_w * E. Drains priming P by commit_cost."""
        E_scalar = float(state.E.mean().item())
        scale = eta_w * E_scalar
        for name, p in named_params.items():
            if name in grads_safe and p.requires_grad:
                p.data.add_(grads_safe[name], alpha=-scale)
        state.P = torch.clamp(state.P - commit_cost, 0.0, 1.0)
        state.commits += 1
