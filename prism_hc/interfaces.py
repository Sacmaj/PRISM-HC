"""Typed interfaces for PRISM-HC phase 2/3 module boundaries."""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch

from .state import ControllerState, ReservoirState, SafetyState


@runtime_checkable
class PlasticityController(Protocol):
    """Controller API implemented by LATCHPlasticityController."""

    def step(
        self,
        state: ControllerState,
        safety: SafetyState,
        surprise_drive: torch.Tensor,
        dt: float = 1.0,
    ) -> ControllerState:
        ...

    def can_commit(self, state: ControllerState, P_min: float) -> bool:
        ...

    def diagnose(self, state: ControllerState, P_min: float) -> str:
        ...


@runtime_checkable
class SafetyAnchorCore(Protocol):
    """Safety measurement API implemented by FrozenAnchorCore."""

    def compute_coords(self, h_dict: Dict[int, torch.Tensor]) -> SafetyState:
        ...

    def drift_loss(
        self, h_new: torch.Tensor, h_ref: torch.Tensor
    ) -> torch.Tensor:
        ...


@runtime_checkable
class Router(Protocol):
    """Reservoir routing API implemented by SeededBilinearReservoir."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def route(
        self,
        x: torch.Tensor,
        route_state: ReservoirState,
        top_k: Optional[int],
        prune_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def update_health(
        self,
        route_state: ReservoirState,
        reward: torch.Tensor,
        decay: float,
        reward_rate: float,
        floor: float,
    ) -> ReservoirState:
        ...
