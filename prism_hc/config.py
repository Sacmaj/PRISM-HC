"""Static configuration constants for PRISM-HC-lite.

All time constants are per-step (dt=1.0 by default).
All gating thresholds and CBF parameters live here so the model has
a single source of truth and can be re-tuned without code edits.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


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
    lam_chi: float = 0.05

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

    @classmethod
    def from_rebus_synthesis(
        cls,
        gains: Any,
        *,
        gamma_map: Optional[Callable[[Any], Tuple[float, float, float]]] = None,
        **overrides: Any,
    ) -> "PrismConfig":
        """Build PrismConfig from REBUS supervisor-gain synthesis output.

        `gains` is duck-typed: it must expose `.p`, `.q`, `.delta_safe` (and
        optionally `.Gamma`) attributes, matching the SupervisorGains dataclass
        in AI Papers/rebus_identification.py.

        The default mapping is a HEURISTIC, not a derivation. The synthesizer's
        composite-Lyapunov coefficients (p, q, delta_safe, Gamma) and the
        REBUS-update forcing-term coefficients (gamma_s, gamma_d, gamma_eps)
        are mathematically distinct objects. Default:

            (gamma_s, gamma_d, gamma_eps) = (0.5/p, 0.5/q, 0.5*delta_safe)

        Override via `gamma_map=lambda g: (...)` if you have a principled
        relationship in mind. `Gamma` has no current home in PrismConfig and
        is dropped on the floor; revisit if/when supervisor-gain enforcement
        gets wired into LATCH.
        """
        if gamma_map is None:
            gamma_map = lambda g: (0.5 / g.p, 0.5 / g.q, 0.5 * g.delta_safe)
        gs, gd, ge = gamma_map(gains)
        return dataclasses.replace(
            cls(),
            gamma_s=float(gs),
            gamma_d=float(gd),
            gamma_eps=float(ge),
            **overrides,
        )
