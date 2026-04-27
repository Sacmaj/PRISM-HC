"""Static configuration constants for PRISM-HC-lite.

All time constants are per-step (dt=1.0 by default).
All gating thresholds and CBF parameters live here so the model has
a single source of truth and can be re-tuned without code edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union


def default_gamma_map(gains: Any) -> Tuple[float, float, float]:
    """Heuristic mapping from synthesizer Lyapunov coefficients to R-dynamics gammas.

    Returns ``(gamma_s, gamma_d, gamma_eps) = (0.5/p, 0.5/q, 0.5*delta_safe)``
    where ``(p, q, delta_safe)`` come from
    :func:`rebus_synthesis.identification.synthesize_supervisor_gains`.

    This is a HEURISTIC, not a derivation. The two coefficient sets are
    mathematically distinct objects:

    - ``(p, q, delta_safe)`` are composite-Lyapunov certificate coefficients
      from the S-procedure / LMI synthesis at
      ``rebus_synthesis/identification.py:1005-1036``.
    - ``(gamma_s, gamma_d, gamma_eps)`` are parameters of the R-dynamics ODE
      in ``AI Papers/rebus_control_framework.tex`` eq. (R_dynamics, line 161)
      and the majorant ``nu(t)`` (line 202). They are *inputs* to the model,
      not derivable from the synthesizer's Lyapunov bounds.

    The framework provides no inverse map. This default exists only as a
    smoke-test starting point that scales with certificate tightness; pass
    ``gamma_map=...`` to :meth:`PrismConfig.from_rebus_synthesis` for any
    use beyond exercising the wire-up.

    ``gains`` is duck-typed: must expose ``.p``, ``.q``, ``.delta_safe``.
    Caller's contract: ``p > 0`` and ``q > 0`` (guaranteed by
    ``SupervisorGains`` LMI feasibility — no zero-guard here).
    """
    return (0.5 / gains.p, 0.5 / gains.q, 0.5 * gains.delta_safe)


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
    # Robustness offset added to cbf_delta inside LATCH:
    # delta_eff = cbf_delta + cbf_robust_gamma. Default 0.0 preserves
    # bit-identical behavior; from_rebus_synthesis populates this from
    # SupervisorGains.Gamma so the synthesizer's disturbance-gain bound
    # actually shrinks LATCH's safe set.
    cbf_robust_gamma: float = 0.0

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

    # Precision modulation per-layer. Accepts a scalar (broadcast to all L
    # layers), a 1-tuple (broadcast), or a tuple of length L; normalized to a
    # length-L tuple by __post_init__.
    delta_l: Union[float, Tuple[float, ...]] = 0.30
    kappa_l: Union[float, Tuple[float, ...]] = 0.50
    log_pi_clamp: Tuple[float, float] = (-6.0, 6.0)

    def __post_init__(self) -> None:
        self.delta_l = self._normalize_per_layer("delta_l", self.delta_l)
        self.kappa_l = self._normalize_per_layer("kappa_l", self.kappa_l)
        # cbf_robust_gamma is an additive robustness offset on top of
        # cbf_delta. A negative value would shrink delta_eff below baseline
        # and expand the CBF safe set, inverting the intended robustness
        # tightening (e.g. a state at S=0.20, E=0.60 that fails commit at
        # gamma=0 would pass at gamma=-0.20). Upstream Gamma = b3_ub + p*eta
        # can go negative if a caller passes negative eta; refuse loudly
        # rather than silently flipping the safety semantics.
        if self.cbf_robust_gamma < 0.0:
            raise ValueError(
                f"cbf_robust_gamma must be >= 0 (got {self.cbf_robust_gamma}); "
                f"a negative robustness offset would expand the CBF safe set "
                f"and invert the intended tightening."
            )
        # Joint-CBF safe set is {S in [0,1] : S - cbf_a*E^p >= delta_eff} where
        # delta_eff = cbf_delta + cbf_robust_gamma; S is clamped to [0,1] in
        # latch.py:67. If delta_eff >= 1, no S can satisfy the bound, so the
        # safe set is empty and LATCH can never commit. This slipped through
        # silently before: a misconverged synthesizer producing a runaway
        # `Gamma` would build a config that refuses every action with no
        # diagnostic. Surface it at construction instead.
        delta_eff = self.cbf_delta + self.cbf_robust_gamma
        if delta_eff >= 1.0:
            raise ValueError(
                f"cbf_delta + cbf_robust_gamma must be < 1 (got "
                f"{self.cbf_delta} + {self.cbf_robust_gamma} = {delta_eff}); "
                f"the runtime joint-CBF safe set "
                f"{{S in [0,1] : S - cbf_a*E^p >= delta_eff}} would be empty, "
                f"so LATCH could never commit. Most likely cause: the upstream "
                f"synthesizer emitted a misconverged Gamma."
            )

    def _normalize_per_layer(
        self, name: str, value: Union[float, Tuple[float, ...]]
    ) -> Tuple[float, ...]:
        if isinstance(value, (int, float)):
            return (float(value),) * self.L
        value = tuple(float(v) for v in value)
        if len(value) == 1:
            return value * self.L
        if len(value) != self.L:
            raise ValueError(
                f"{name} has length {len(value)} but L={self.L}; "
                f"pass a scalar, a 1-tuple, or a tuple of length L."
            )
        return value

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
        in rebus_synthesis.identification.

        By default, ``(gamma_s, gamma_d, gamma_eps)`` are derived via
        :func:`default_gamma_map` (a documented heuristic — see its docstring
        for the framework gap). Pass ``gamma_map=...`` to override.

        `Gamma` (read via getattr so duck-typed gains without it still work)
        is the upper bound on disturbance-driven Lyapunov increase under the
        composite Lyapunov function. It propagates into `cbf_robust_gamma`,
        which LATCH adds to `cbf_delta` to shrink the joint CBF safe set
        proportional to the disturbance gain. Pass `cbf_robust_gamma=...`
        in **overrides to scale or zero this contribution at the call site.
        """
        if gamma_map is None:
            gamma_map = default_gamma_map
        gs, gd, ge = gamma_map(gains)
        # Pop so an explicit `cbf_robust_gamma` in overrides wins over the
        # gains-derived value without colliding on the keyword argument.
        gamma_val = float(
            overrides.pop("cbf_robust_gamma", getattr(gains, "Gamma", 0.0))
        )
        # Build directly via cls() rather than dataclasses.replace(cls(), ...):
        # the latter would copy already-normalized length-L=2 tuples for
        # delta_l/kappa_l from the seed instance, then trip __post_init__'s
        # length check if `overrides` raises L without also overriding
        # delta_l/kappa_l.
        return cls(
            gamma_s=float(gs),
            gamma_d=float(gd),
            gamma_eps=float(ge),
            cbf_robust_gamma=gamma_val,
            **overrides,
        )
