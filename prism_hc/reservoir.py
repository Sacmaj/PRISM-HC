"""Seeded bilinear reservoir + linear readout.

phi(x) = einsum('bi,bj,ijo->bo', x, x, W_rand)        # frozen, gradient-free
out(x) = readout(phi(x))                              # only trainable weight

W_rand is a (d_in, d_in, d_out) tensor sampled once with a fixed seed and
registered as a buffer so it never receives gradients. The readout is a
plain Linear so adapter commits flow into it after null-space projection.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .state import ReservoirState


class SeededBilinearReservoir(nn.Module):
    def __init__(self, d_in: int, d_reservoir: int, d_hidden: int, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        scale = 1.0 / max(d_in, 1) ** 0.5
        W = torch.randn(d_in, d_in, d_reservoir, generator=gen) * scale
        self.register_buffer("W_rand", W)
        self.readout = nn.Linear(d_reservoir, d_hidden, bias=False)

    @staticmethod
    def init_state(
        d_reservoir: int,
        device: torch.device,
        dtype: torch.dtype,
        batch: int = 1,
    ) -> ReservoirState:
        return ReservoirState(
            route_health=torch.ones(d_reservoir, device=device, dtype=dtype),
            hits=torch.zeros(d_reservoir, device=device, dtype=dtype),
            active_paths=torch.zeros(
                batch, d_reservoir, device=device, dtype=torch.bool
            ),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,bj,ijo->bo", x, x, self.W_rand)

    @staticmethod
    def select_active(
        phi: torch.Tensor,
        route_state: ReservoirState,
        top_k: Optional[int],
        prune_threshold: float,
    ) -> torch.Tensor:
        health = route_state.route_health.to(device=phi.device, dtype=phi.dtype)
        if top_k is None:
            return torch.ones_like(phi, dtype=torch.bool)

        eligible = health > prune_threshold
        k = min(int(top_k), phi.shape[-1])
        scores = phi.detach().abs() * health.unsqueeze(0)
        scores = scores.masked_fill(
            ~eligible.unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        indices = scores.topk(k=k, dim=-1).indices
        mask = torch.zeros_like(phi, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask & eligible.unsqueeze(0)

    @staticmethod
    def route_entropy(phi: torch.Tensor, active_paths: torch.Tensor) -> torch.Tensor:
        weights = phi.detach().abs() * active_paths.to(phi.dtype)
        totals = weights.sum(dim=-1, keepdim=True)
        probs = torch.where(totals > 0.0, weights / totals.clamp_min(1e-12), weights)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        return entropy.mean()

    def route(
        self,
        x: torch.Tensor,
        route_state: ReservoirState,
        top_k: Optional[int],
        prune_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.features(x)
        active_paths = self.select_active(phi, route_state, top_k, prune_threshold)
        route_state.active_paths = active_paths.detach()
        return self.readout(phi * active_paths.to(phi.dtype)), phi, active_paths

    @staticmethod
    @torch.no_grad()
    def update_health(
        route_state: ReservoirState,
        reward: torch.Tensor,
        decay: float,
        reward_rate: float,
        floor: float,
    ) -> ReservoirState:
        health = route_state.route_health
        active = route_state.active_paths.to(device=health.device, dtype=health.dtype)
        if active.dim() == 2:
            active_rate = active.mean(dim=0)
            hit_increment = active.sum(dim=0)
        else:
            active_rate = active
            hit_increment = active
        reward_value = torch.as_tensor(
            reward, device=health.device, dtype=health.dtype
        ).mean()
        updated = health * (1.0 - decay) + reward_rate * reward_value * active_rate
        route_state.route_health = torch.clamp(updated, min=floor, max=1.0)
        route_state.hits = route_state.hits + hit_increment
        return route_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x))
