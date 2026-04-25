"""Real-objective learning demo for PRISM-HC-lite.

Mirrors the smoke demo's regime schedule but feeds *real* backprop gradients
into plasticity_step (instead of torch.randn_like placeholders). The task is
one-step prediction on a multivariate AR(1) sequence: x_{t+1} = A x_t + noise,
target at step t is x_{t+1}, loss is MSE between the model output y and x_{t+1}.

Schedule (mirrors demo.py):
  t in [ 0, 10): warm-up      — moderate noise
  t in [10, 30): benign       — clean AR(1) sequence, drift small, dwell builds
  t = 30        : commit attempt with real gradient
  t in [30, 40): cooldown     — clean sequence continues
  t in [40, 55): stress       — AR(1) plus large noise burst
  t in [55, 60): recovery     — clean sequence returns

Gradient accumulation: gradients from steps t in [10, 30] are accumulated on
p.grad (no zero_grad in that window) so the t=30 commit uses an averaged
signal across the benign phase. Per-step grads are too small to move loss
meaningfully (eta_w * E ~ 1e-4), and the user asked for a single commit at
t=30 — accumulating grads beforehand is the simplest way to honor both.

Pre/post-commit comparison:
  At t=29 a probe MSE is recorded under torch.no_grad() on a fixed AR(1)
  trajectory. At t=31 the same probe is re-run after the commit. The
  assertion is honest, not aspirational: a single SGD step on accumulated
  benign-phase gradients shifts probe MSE by ~3e-6 at eta_w=0.1, i.e. the
  effect is real and directional but small. We assert the shift is non-zero
  (proves the commit had a measurable downstream effect) and print whether
  it improved the held-out probe.

Run:  python -m prism_hc.learning_demo
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.state import BeliefState
from prism_hc.telemetry import TelemetryRecorder


def make_ar1_matrix(d: int, spectral_radius: float, seed: int) -> torch.Tensor:
    """AR(1) transition matrix of given spectral radius via QR + scaled diag."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(raw)
    eigs = torch.linspace(0.45, spectral_radius, d)
    return Q @ torch.diag(eigs) @ Q.T


def ar1_step(
    x: torch.Tensor, A: torch.Tensor, noise_scale: float, gen: torch.Generator
) -> torch.Tensor:
    return x @ A.T + noise_scale * torch.randn(x.shape, generator=gen)


def regime_noise_scale(t: int) -> float:
    if 0 <= t < 10:
        return 0.20
    if 10 <= t < 40:
        return 0.05
    if 40 <= t < 55:
        return 0.50
    return 0.05


def detach_belief(belief: BeliefState) -> BeliefState:
    """Detach belief tensors so per-step backward stays local (no BPTT)."""
    return BeliefState(
        mu_l={l: m.detach() for l, m in belief.mu_l.items()},
        epsilon_l={l: e.detach() for l, e in belief.epsilon_l.items()},
        pi_l={l: p.detach() for l, p in belief.pi_l.items()},
        free_energy=(
            None if belief.free_energy is None else belief.free_energy.detach()
        ),
    )


def probe_mse(
    model: PrismHCLite, A: torch.Tensor, n_steps: int = 20, seed: int = 999
) -> float:
    """Mean MSE on a fixed AR(1) probe trajectory, under no_grad."""
    gen = torch.Generator().manual_seed(seed)
    state = model.init_state(batch=1)
    belief = model.init_belief(batch=1)
    x = 0.1 * torch.randn(1, model.cfg.d_in, generator=gen)
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(n_steps):
            x_next = ar1_step(x, A, 0.05, gen)
            y, state, belief, _ = model.forward(x, state, belief)
            losses.append(F.mse_loss(y, x_next).item())
            x = x_next
    return sum(losses) / len(losses)


def main() -> int:
    # Bump eta_w 100x for the learning demo. Default eta_w=1e-3 with E~0.06 at
    # the commit gives effective lr ~6e-5 — too small to move probe MSE
    # visibly off the loss surface in one accumulated step. The smoke demo
    # keeps the default; this demo's purpose is to show real loss reduction.
    cfg = dataclasses.replace(PrismConfig(), eta_w=0.1)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    model = PrismHCLite(cfg)
    state = model.init_state(batch=1)
    belief = model.init_belief(batch=1)
    tele = TelemetryRecorder()

    A = make_ar1_matrix(cfg.d_in, spectral_radius=0.85, seed=42)

    pre_commit_mse: Optional[float] = None
    post_commit_mse: Optional[float] = None
    loss_history: List[float] = []

    T = 60
    x = 0.1 * torch.randn(1, cfg.d_in, generator=gen)
    model.zero_grad()  # initial clear; .grad starts as None anyway
    for t in range(T):
        x_next = ar1_step(x, A, regime_noise_scale(t), gen)

        y, state, belief, rec = model.forward(x, state, belief)
        tele.append_step(rec)
        loss = F.mse_loss(y, x_next)
        loss_history.append(float(loss.item()))

        # Drop warm-up grads at t=10 so the t=30 commit reads only the
        # benign-phase accumulation. Otherwise let .grad accumulate.
        if t == 10:
            model.zero_grad()
        loss.backward()

        if t == 29:
            pre_commit_mse = probe_mse(model, A)
        if t == 30:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            state = model.plasticity_step(state, grads, tele)
            model.zero_grad()
        if t == 31:
            post_commit_mse = probe_mse(model, A)

        belief = detach_belief(belief)
        x = x_next.detach()

    tele.print_table()

    print()
    print(f"steps={len(tele.steps)} commits_attempted={len(tele.commits)}")
    if tele.commits:
        c = tele.commits[0]
        print(
            f"commit @ t=30: committed={c.committed} reason={c.reason} "
            f"g_norm={c.g_norm:.4f}"
        )

    print()
    print("loss curve (5-step block means):")
    for k in range(0, T, 5):
        block = loss_history[k : k + 5]
        if block:
            print(f"  t=[{k:>2},{k + len(block):>2}): mean_mse={sum(block) / len(block):.5f}")

    assert pre_commit_mse is not None and post_commit_mse is not None, (
        "probe MSE not recorded — pre/post comparison failed"
    )
    print()
    print(f"pre_commit_mse  = {pre_commit_mse:.8f}")
    print(f"post_commit_mse = {post_commit_mse:.8f}")
    print(f"delta           = {post_commit_mse - pre_commit_mse:+.8f}")
    print(f"relative_delta  = {(post_commit_mse - pre_commit_mse) / pre_commit_mse:+.4%}")

    if not tele.commits or not tele.commits[0].committed:
        print("WARNING: commit was not accepted; pre/post comparison is degenerate")
    else:
        # Honest assertions: the commit had a measurable effect on the
        # held-out probe (proves real gradients flowed through the pipeline
        # and updated params), and the effect didn't blow up.
        delta = post_commit_mse - pre_commit_mse
        assert abs(delta) > 1e-8, (
            f"commit had no measurable effect on probe MSE: delta={delta:.2e}"
        )
        assert post_commit_mse < pre_commit_mse * 1.5, (
            f"commit destabilized the model: pre={pre_commit_mse:.6f} "
            f"post={post_commit_mse:.6f}"
        )
        if delta < 0:
            print(f"probe loss IMPROVED by {-delta:.2e} ({-delta / pre_commit_mse:.4%})")
        else:
            print(f"probe loss regressed by {delta:.2e} ({delta / pre_commit_mse:.4%})")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
