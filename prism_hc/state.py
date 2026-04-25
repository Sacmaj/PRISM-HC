"""Typed state primitives for PRISM-HC-lite.

Ported from AI Papers/gemini-code-1777060060070.py with two additions:
  - ControllerState.dwell_counter (long tensor) — supports the LATCH
    lexicographic rule by counting consecutive steps with S >= S_min.
  - ControllerState.commits (int) — running tally of accepted commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class BeliefState:
    """Hierarchical predictive-processing state for a single forward pass."""
    mu_l: Dict[int, torch.Tensor]
    epsilon_l: Dict[int, torch.Tensor]
    pi_l: Dict[int, torch.Tensor]
    free_energy: Optional[torch.Tensor] = None


@dataclass
class TopologyState:
    """Routing-graph bookkeeping. Unused in lite pass; reserved for growth/prune."""
    active_edges: int
    topology_mass: float
    birth_budget: float
    prune_budget: float


@dataclass
class SafetyState:
    """Measurements from the frozen cross-modal anchor core."""
    anchor_coords: Dict[int, torch.Tensor]
    drift: torch.Tensor
    canary_margin: float
    veto_logits: Optional[torch.Tensor] = None


@dataclass
class ReservoirState:
    """State for the gradient-free seeded tensor router. Reserved for Hebbian."""
    route_health: torch.Tensor
    hits: torch.Tensor
    active_paths: torch.Tensor


@dataclass
class ControllerState:
    """LATCH + REBUS governor state — the system's central API boundary."""
    R_l: Dict[int, torch.Tensor]
    alpha: torch.Tensor
    h: torch.Tensor

    E: torch.Tensor
    S: torch.Tensor
    rho: torch.Tensor
    chi: torch.Tensor
    P: torch.Tensor

    dwell_counter: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.long)
    )
    commits: int = 0
