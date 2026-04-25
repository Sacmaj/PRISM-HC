"""Static configuration constants for PRISM-HC-lite.

All time constants are per-step (dt=1.0 by default).
All gating thresholds and CBF parameters live here so the model has
a single source of truth and can be re-tuned without code edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PrismConfig:
    # Topology
    L: int = 2
    d_in: int = 8
    d_hidden: int = 16
    d_reservoir: int = 32
    n_anchors: int = 4

    # Integration
    dt: float = 1.0
    seed: int = 0

    # LATCH dual-state time constants
    lam_E: float = 0.10
    lam_S: float = 0.15
    lam_rho: float = 0.05

    # LATCH gating thresholds
    S_min: float = 0.7
    dwell_min: int = 3
    rho_max: float = 0.9

    # Joint CBF h(E,S) = S - a*E^p - delta >= 0
    cbf_a: float = 0.5
    cbf_p: float = 2.0
    cbf_delta: float = 0.05

    # Priming / commit
    P_min: float = 0.6
    commit_cost: float = 0.3
    eta_w: float = 1e-3

    # REBUS R update
    lam_R: float = 0.08
    R0: float = 0.0
    beta_h: float = 0.10
    gamma_s: float = 0.20
    gamma_d: float = 0.15
    gamma_eps: float = 0.10

    # Homeostatic budget h
    lam_h: float = 0.03
    eta_h: float = 0.05

    # Precision modulation per-layer
    delta_l: Tuple[float, ...] = (0.30, 0.30)
    kappa_l: Tuple[float, ...] = (0.50, 0.50)
    log_pi_clamp: Tuple[float, float] = (-6.0, 6.0)
