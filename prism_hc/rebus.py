"""REBUS scalar-state updates and precision modulation.

Discrete-time exponential-Euler integration of:
    dS/dt = -lam * (S - target)              -> exp_euler(target=...)
    dR/dt = -lam_R*(R-R0) + drive(state)     -> update_R
    dh/dt = -lam_h*h + eta_h*ReLU(R-R0)      -> update_h
    Pi_l = Pi_bar_l * exp(delta_l*u_d - kappa_l*R)   -> precision_modulation

The exp_euler form treats 'target' as the steady-state value of the linear ODE
    dx/dt = -lam * (x - target)
giving the exact discrete update:
    x_{t+dt} = x_t * exp(-lam*dt) + (1 - exp(-lam*dt)) * target
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def exp_euler(
    current: torch.Tensor,
    target: torch.Tensor,
    lam: float,
    dt: float = 1.0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> torch.Tensor:
    """Asymptote-form exponential Euler. `target` is the steady-state value."""
    leak = math.exp(-lam * dt)
    nxt = current * leak + (1.0 - leak) * target
    return torch.clamp(nxt, lo, hi)


def precision_modulation(
    pi_bar: torch.Tensor,
    R: torch.Tensor,
    u_d: torch.Tensor,
    delta_l: float,
    kappa_l: float,
    log_clamp: Tuple[float, float] = (-6.0, 6.0),
) -> torch.Tensor:
    """Pi_l(R, u_d) = Pi_bar_l * exp( clamp(delta*u_d - kappa*R) ).

    Computed in log-space and clamped to keep precision finite and positive.
    Broadcasts pi_bar over batch via tensor broadcasting at the call site.
    """
    log_pi = torch.log(pi_bar) + delta_l * u_d - kappa_l * R
    return torch.exp(log_pi.clamp(min=log_clamp[0], max=log_clamp[1]))


@torch.no_grad()
def update_R(
    R: torch.Tensor,
    h: torch.Tensor,
    u_s: torch.Tensor,
    u_d: torch.Tensor,
    eps_norm: torch.Tensor,
    lam_R: float,
    R0: float,
    beta_h: float,
    gamma_s: float,
    gamma_d: float,
    gamma_eps: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """Discrete REBUS R update around R0.

    drive = -beta_h*h + gamma_s*sigmoid(u_s) + gamma_d*tanh(u_d) + gamma_eps*tanh(eps_norm)
    Continuous form: dR/dt = -lam_R*(R-R0) + drive.
    Steady-state target: R0 + drive/lam_R. We use exp-Euler with that target,
    then clamp to [0, 1].
    """
    drive = (
        -beta_h * h
        + gamma_s * torch.sigmoid(u_s)
        + gamma_d * torch.tanh(u_d)
        + gamma_eps * torch.tanh(eps_norm)
    )
    target = R0 + drive / max(lam_R, 1e-6)
    return exp_euler(R, target, lam_R, dt, lo=0.0, hi=1.0)


@torch.no_grad()
def update_h(
    h: torch.Tensor,
    R: torch.Tensor,
    R0: float,
    lam_h: float,
    eta_h: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """Homeostatic budget. dh/dt = -lam_h*h + eta_h*ReLU(R-R0)."""
    drive = eta_h * torch.relu(R - R0)
    target = drive / max(lam_h, 1e-6)
    return exp_euler(h, target, lam_h, dt, lo=0.0, hi=1.0)
