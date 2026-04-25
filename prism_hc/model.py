"""PrismHCLite: assembles all modules into one forward + plasticity loop.

Forward order (matches Section 3 of the plan):
  (1) bottom_up h_dict
  (2) FrozenAnchorCore.compute_coords -> SafetyState
  (3) LATCH.step -> S, dwell, E, rho, CBF clamp
  (4) precision_modulation per layer
  (5) reservoir routing + PGSTR gate -> mu_l
  (6) errors, free-energy
  (7) REBUS update_R per layer, homeostat update_h
  (8) telemetry record

Plasticity step (post-forward):
  - LATCH.can_commit gate
  - null-space project gradient
  - SGD step scaled by eta_w * E
  - drain priming P
  - record commit
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from . import rebus
from .adapter import NullSpaceAdapter
from .anchors import FrozenAnchorCore
from .config import PrismConfig
from .hierarchy import BeliefHierarchy
from .latch import LATCHPlasticityController
from .reservoir import SeededBilinearReservoir
from .routing import pgstr_gate
from .state import BeliefState, ControllerState, SafetyState
from .telemetry import CommitRecord, StepRecord, TelemetryRecorder


class PrismHCLite(nn.Module):
    def __init__(self, cfg: PrismConfig):
        super().__init__()
        self.cfg = cfg
        self.anchors = FrozenAnchorCore(cfg.d_hidden, cfg.n_anchors, cfg.seed)
        self.hierarchy = BeliefHierarchy(cfg)
        self.reservoir = SeededBilinearReservoir(
            cfg.d_in, cfg.d_reservoir, cfg.d_hidden, cfg.seed
        )
        self.latch = LATCHPlasticityController(
            lam_E=cfg.lam_E,
            lam_S=cfg.lam_S,
            lam_rho=cfg.lam_rho,
            S_min=cfg.S_min,
            dwell_min=cfg.dwell_min,
            rho_max=cfg.rho_max,
            cbf_a=cfg.cbf_a,
            cbf_p=cfg.cbf_p,
            cbf_delta=cfg.cbf_delta,
        )
        self.register_buffer("pi_bar", torch.ones(cfg.L, cfg.d_hidden))

    # ---- state init ------------------------------------------------------

    def init_state(self, batch: int = 1) -> ControllerState:
        z = lambda: torch.zeros(batch)
        return ControllerState(
            R_l={l: z() for l in range(self.cfg.L)},
            alpha=z(),
            h=z(),
            E=z(),
            S=z(),
            rho=z(),
            chi=z(),
            P=torch.ones(batch),
            dwell_counter=torch.zeros(batch, dtype=torch.long),
            commits=0,
        )

    def init_belief(self, batch: int = 1) -> BeliefState:
        z = lambda: torch.zeros(batch, self.cfg.d_hidden)
        return BeliefState(
            mu_l={l: z() for l in range(self.cfg.L)},
            epsilon_l={l: z() for l in range(self.cfg.L)},
            pi_l={l: torch.ones(batch, self.cfg.d_hidden) for l in range(self.cfg.L)},
        )

    def reset_episode(self, state: ControllerState) -> ControllerState:
        for l in state.R_l:
            state.R_l[l].zero_()
        state.alpha.zero_()
        state.h.zero_()
        state.E.zero_()
        state.S.zero_()
        state.rho.zero_()
        state.chi.zero_()
        state.P.fill_(1.0)
        state.dwell_counter.zero_()
        return state

    # ---- forward ---------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        state: ControllerState,
        belief_prev: BeliefState,
    ) -> Tuple[torch.Tensor, ControllerState, BeliefState, StepRecord]:
        cfg = self.cfg

        # (1) bottom-up + (2) anchor coords / drift
        h_dict = self.hierarchy.bottom_up(x)
        safety: SafetyState = self.anchors.compute_coords(h_dict)

        # (3) LATCH dual-state step (S, dwell, E, rho, CBF clamp)
        surprise_drive = torch.tanh(safety.drift.detach()).expand_as(state.E)
        state = self.latch.step(state, safety, surprise_drive, dt=cfg.dt)

        # (4) per-layer precision Pi_l(R_l, u_d=E)
        pi_l: Dict[int, torch.Tensor] = {}
        E_b = state.E.unsqueeze(-1)  # (batch, 1)
        for l in range(cfg.L):
            R_b = state.R_l[l].unsqueeze(-1)
            pi_l[l] = rebus.precision_modulation(
                self.pi_bar[l], R_b, E_b,
                delta_l=cfg.delta_l[l],
                kappa_l=cfg.kappa_l[l],
                log_clamp=cfg.log_pi_clamp,
            )

        # (5) reservoir-routed features + PGSTR mixing per layer
        routed = self.reservoir(x)
        mu_l: Dict[int, torch.Tensor] = {}
        pred_l: Dict[int, torch.Tensor] = {}
        for l in range(cfg.L):
            prior = belief_prev.mu_l[l]
            evidence = h_dict[l]
            routed_l = routed if l == 0 else torch.zeros_like(evidence)
            R_b = state.R_l[l].unsqueeze(-1)
            mu_l[l] = pgstr_gate(prior, evidence, routed_l, R_b)
            pred_l[l] = self.hierarchy.predict(l, mu_l[l])

        # (6) errors and free-energy
        eps_l = {l: h_dict[l] - pred_l[l] for l in range(cfg.L)}
        F = self.hierarchy.free_energy(pi_l, eps_l)
        belief = BeliefState(mu_l=mu_l, epsilon_l=eps_l, pi_l=pi_l, free_energy=F)

        # (7) REBUS R update per layer, homeostat h
        eps_norms = torch.stack([e.norm(dim=-1).mean() for e in eps_l.values()])
        eps_norm = eps_norms.mean().expand_as(state.E)
        for l in range(cfg.L):
            state.R_l[l] = rebus.update_R(
                state.R_l[l],
                h=state.h,
                u_s=state.S,
                u_d=state.E,
                eps_norm=eps_norm,
                lam_R=cfg.lam_R,
                R0=cfg.R0,
                beta_h=cfg.beta_h,
                gamma_s=cfg.gamma_s,
                gamma_d=cfg.gamma_d,
                gamma_eps=cfg.gamma_eps,
                dt=cfg.dt,
            )
        state.h = rebus.update_h(
            state.h, state.R_l[0], R0=cfg.R0,
            lam_h=cfg.lam_h, eta_h=cfg.eta_h, dt=cfg.dt,
        )
        state.chi = rebus.exp_euler(
            state.chi, torch.zeros_like(state.chi),
            cfg.lam_chi, cfg.dt, lo=0.0, hi=1.0,
        )

        # (8) telemetry
        cbf = state.S - cfg.cbf_a * state.E.pow(cfg.cbf_p) - cfg.cbf_delta
        rec = StepRecord(
            R=float(state.R_l[0].mean().item()),
            E=float(state.E.mean().item()),
            S=float(state.S.mean().item()),
            rho=float(state.rho.mean().item()),
            chi=float(state.chi.mean().item()),
            P=float(state.P.mean().item()),
            h=float(state.h.mean().item()),
            F=float(F.detach().item()),
            drift=float(safety.drift.detach().item()),
            cbf=float(cbf.mean().item()),
            dwell=int(state.dwell_counter.min().item()),
        )

        y = self.hierarchy.readout(mu_l[cfg.L - 1])
        return y, state, belief, rec

    # ---- plasticity -------------------------------------------------------

    def plasticity_step(
        self,
        state: ControllerState,
        grads: Dict[str, torch.Tensor],
        telemetry: TelemetryRecorder,
    ) -> ControllerState:
        cfg = self.cfg
        if not self.latch.can_commit(state, cfg.P_min):
            telemetry.append_commit(
                CommitRecord(
                    committed=False,
                    reason=self.latch.diagnose(state, cfg.P_min),
                    g_norm=0.0,
                )
            )
            return state

        U = self.anchors.U
        g_safe = {
            name: NullSpaceAdapter.project_update(g, U) for name, g in grads.items()
        }
        named = dict(self.named_parameters())
        NullSpaceAdapter.apply_commit(
            state, named, g_safe, cfg.eta_w, cfg.commit_cost
        )
        g_norm = float(sum(g.norm().item() for g in g_safe.values()))
        telemetry.append_commit(
            CommitRecord(committed=True, reason="ok", g_norm=g_norm)
        )
        return state
