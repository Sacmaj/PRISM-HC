"""LATCH dual-state controller (E, S, rho).

Lexicographic rule: safety must establish before plasticity can rise.

  S[t+1] = exp_euler(S, target = 1 - drift, lam_S, dt)
  dwell  = dwell + 1 if S >= S_min else 0
  gate   = (dwell >= dwell_min) AND (rho < rho_max)
  E[t+1] = exp_euler(E, target = surprise * gate, lam_E, dt)
  rho[t+1] = exp_euler(rho, target = E_next, lam_rho, dt)
  E[t+1] <- min(E[t+1], sqrt(clamp((S - delta) / a, 0)))   # joint CBF clamp

Joint CBF: h(E, S) = S - a * E^p - delta >= 0 carves out the safe set in (E,S).

Ported and corrected from AI Papers/gemini-code-1777060065446.py:
  - exp_euler now takes the steady-state TARGET (asymptote form) instead of
    raw forcing — matches the spec interpretation `dS/dt = -lam(S - target)`.
  - CBF parameters (a, p, delta) are now constructor args rather than hard-
    coded in the formula.
  - `can_commit` exposes the joint commit-eligibility predicate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .state import ControllerState, SafetyState


class LATCHPlasticityController:
    def __init__(
        self,
        lam_E: float = 0.10,
        lam_S: float = 0.15,
        lam_rho: float = 0.05,
        S_min: float = 0.7,
        dwell_min: int = 3,
        rho_max: float = 0.9,
        cbf_a: float = 0.5,
        cbf_p: float = 2.0,
        cbf_delta: float = 0.05,
    ):
        self.lam_E = lam_E
        self.lam_S = lam_S
        self.lam_rho = lam_rho
        self.S_min = S_min
        self.dwell_min = dwell_min
        self.rho_max = rho_max
        self.cbf_a = cbf_a
        self.cbf_p = cbf_p
        self.cbf_delta = cbf_delta

    @staticmethod
    def _exp_euler(
        current: torch.Tensor,
        target: torch.Tensor,
        lam: float,
        dt: float,
    ) -> torch.Tensor:
        leak = math.exp(-lam * dt)
        nxt = current * leak + (1.0 - leak) * target
        return torch.clamp(nxt, 0.0, 1.0)

    @torch.no_grad()
    def step(
        self,
        state: "ControllerState",
        safety: "SafetyState",
        surprise_drive: torch.Tensor,
        dt: float = 1.0,
    ) -> "ControllerState":
        # (1) Safety: target = 1 - drift, smoothed.
        instant_safety = torch.clamp(1.0 - safety.drift, 0.0, 1.0)
        instant_safety_b = instant_safety.expand_as(state.S)
        S_next = self._exp_euler(state.S, instant_safety_b, self.lam_S, dt)

        # (2) Lexicographic dwell counter.
        is_safe = S_next >= self.S_min
        state.dwell_counter = torch.where(
            is_safe,
            state.dwell_counter + 1,
            torch.zeros_like(state.dwell_counter),
        )
        safety_authorized = state.dwell_counter >= self.dwell_min
        refractory_ok = state.rho < self.rho_max
        gate = safety_authorized & refractory_ok

        # (3) Entropic openness E. Target is surprise when gated; 0 otherwise.
        gated_target = surprise_drive * gate.to(surprise_drive.dtype)
        E_next = self._exp_euler(state.E, gated_target, self.lam_E, dt)

        # (4) Refractory rho tracks E (capacity accumulator).
        rho_next = self._exp_euler(state.rho, E_next, self.lam_rho, dt)

        # (5) Joint CBF clamp: forbid high E under low S.
        max_safe_E = torch.sqrt(
            torch.clamp((S_next - self.cbf_delta) / max(self.cbf_a, 1e-6), min=0.0)
        )
        E_next = torch.minimum(E_next, max_safe_E)

        state.E = E_next
        state.S = S_next
        state.rho = rho_next
        return state

    @torch.no_grad()
    def can_commit(
        self,
        state: "ControllerState",
        P_min: float,
    ) -> bool:
        """All four eligibility conditions must hold (per batch element)."""
        dwell_ok = state.dwell_counter >= self.dwell_min
        P_ok = state.P >= P_min
        rho_ok = state.rho < self.rho_max
        cbf = state.S - self.cbf_a * state.E.pow(self.cbf_p) - self.cbf_delta
        cbf_ok = cbf >= 0.0
        return bool((dwell_ok & P_ok & rho_ok & cbf_ok).all().item())

    @torch.no_grad()
    def diagnose(self, state: "ControllerState", P_min: float) -> str:
        if not bool((state.dwell_counter >= self.dwell_min).all().item()):
            return "dwell_too_short"
        if not bool((state.P >= P_min).all().item()):
            return "priming_below_threshold"
        if not bool((state.rho < self.rho_max).all().item()):
            return "refractory_saturated"
        cbf = state.S - self.cbf_a * state.E.pow(self.cbf_p) - self.cbf_delta
        if not bool((cbf >= 0.0).all().item()):
            return "cbf_violated"
        return "ok"

    @torch.no_grad()
    def project_cbf(
        self,
        alpha_nominal: torch.Tensor,
        F_alpha: torch.Tensor,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Scalar CBF projection into [alpha_min, alpha_max]. Reserved utility."""
        lower = -F_alpha - gamma * (alpha_nominal - alpha_min)
        upper = -F_alpha + gamma * (alpha_max - alpha_nominal)
        projected = torch.maximum(alpha_nominal, lower)
        projected = torch.minimum(projected, upper)
        return torch.clamp(projected, alpha_min, alpha_max)
