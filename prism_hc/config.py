"""Static configuration constants for PRISM-HC-lite.

All time constants are per-step (dt=1.0 by default).
All gating thresholds and CBF parameters live here so the model has
a single source of truth and can be re-tuned without code edits.

Per-layer arrays (`delta_l`, `kappa_l`) accept a scalar, a length-1
sequence, or a length-`L` sequence. Anything else raises in
`__post_init__` so a mismatched depth fails loudly instead of
producing an `IndexError` mid-forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union


PerLayer = Union[float, Sequence[float]]


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

    # Precision modulation per-layer (scalar or length-L sequence)
    delta_l: PerLayer = 0.30
    kappa_l: PerLayer = 0.50
    log_pi_clamp: Tuple[float, float] = (-6.0, 6.0)

    def __post_init__(self) -> None:
        self.delta_l = self._normalize_per_layer(self.delta_l, "delta_l")
        self.kappa_l = self._normalize_per_layer(self.kappa_l, "kappa_l")

    def _normalize_per_layer(self, val: PerLayer, name: str) -> Tuple[float, ...]:
        if isinstance(val, (int, float)):
            return tuple([float(val)] * self.L)
        seq = tuple(float(v) for v in val)
        if len(seq) == 1:
            return seq * self.L
        if len(seq) == self.L:
            return seq
        raise ValueError(
            f"{name} must be a scalar, length-1 sequence, or length-{self.L} "
            f"sequence to match L; got length {len(seq)}"
        )
