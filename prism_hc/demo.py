"""End-to-end smoke run: 60 steps with one forced plasticity-commit attempt.

Schedule:
  t in [ 0, 10): warm-up — random low-amplitude input
  t in [10, 30): benign — coherent low-noise input, drift small, S rises, dwell builds
  t = 30        : manual plasticity-commit attempt
  t in [30, 40): cooldown — same benign signal
  t in [40, 55): stress — drift spikes, lexicographic gate re-closes
  t in [55, 60): recovery — benign signal returns

Asserts:
  - 0 <= S <= 1, 0 <= E <= 1, 0 <= rho <= rho_max + tol
  - joint CBF margin h(E,S) >= -tol
  - free-energy F is finite (no NaN)
"""

from __future__ import annotations

import math
import sys

import torch

# Allow `python demo.py` from inside this directory by exposing the parent on path.
if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


def synthetic_input(t: int, d_in: int, gen: torch.Generator) -> torch.Tensor:
    """Tiny benign sinusoid (low drift) with a stress burst at t in [40, 55).

    The benign amplitude is deliberately small so that hidden activations
    project mostly inside the random 4-anchor subspace, keeping anchor drift
    well below 1 - S_min. That lets S climb past S_min, dwell accumulate,
    and the t=30 plasticity-commit attempt actually land.
    """
    if 40 <= t < 55:
        # Stress: large random noise plus the original-amplitude sine.
        base = 0.5 * math.sin(t / 3.0) * torch.ones(1, d_in)
        return base + 0.8 * torch.randn(1, d_in, generator=gen)
    # Benign: small coherent signal so drift stays low.
    base = 0.05 * math.sin(t / 3.0) * torch.ones(1, d_in)
    noise = 0.02 * torch.randn(1, d_in, generator=gen)
    return base + noise


def main() -> int:
    cfg = PrismConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    model = PrismHCLite(cfg)
    state = model.init_state(batch=1)
    belief = model.init_belief(batch=1)
    tele = TelemetryRecorder()

    T = 60
    for t in range(T):
        x = synthetic_input(t, cfg.d_in, gen)
        _y, state, belief, rec = model.forward(x, state, belief)
        tele.append_step(rec)
        if t == 30:
            grads = {
                name: torch.randn_like(p, generator=gen) * 0.01
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            state = model.plasticity_step(state, grads, tele)

    tele.print_table()

    # Bound checks
    tol = 1e-5
    rho_max = cfg.rho_max
    for r in tele.steps:
        assert -tol <= r.S <= 1.0 + tol, f"S out of bounds: {r.S}"
        assert -tol <= r.E <= 1.0 + tol, f"E out of bounds: {r.E}"
        assert r.rho <= rho_max + tol, f"rho exceeded rho_max: {r.rho}"
        assert r.cbf >= -tol, f"CBF margin negative: {r.cbf}"
        assert not (r.F != r.F), "free-energy NaN"  # NaN guard

    print()
    print(f"steps={len(tele.steps)} commits_attempted={len(tele.commits)}")
    if tele.commits:
        c = tele.commits[0]
        print(
            f"first commit: t={c.step_index} committed={c.committed} "
            f"reason={c.reason} g_norm={c.g_norm:.4f}"
        )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
