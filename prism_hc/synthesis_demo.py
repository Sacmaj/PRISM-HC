"""End-to-end smoke check: REBUS synthesis scaffold drives the prototype config.

The scaffold lives at `rebus_synthesis/` (sibling package). Two cases:

  1. cvxpy installed: run the full synthesis pipeline
     (make_synthetic_rebus_data -> identify_rebus_bounds ->
     synthesize_supervisor_gains) and feed the gains into PrismConfig.
  2. cvxpy missing: skip the LMI solve and exercise the wire-up with a
     fabricated SupervisorGains-shaped object.

PrismConfig.from_rebus_synthesis duck-types its `gains` argument and
never imports from `rebus_synthesis/`, so the prototype's wire-up
surface is testable without cvxpy via the StubSupervisorGains path.

Run:  python -m prism_hc.synthesis_demo
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


@dataclass(frozen=True)
class StubSupervisorGains:
    """Stand-in for SupervisorGains when cvxpy isn't installed.

    Field names and semantics match SupervisorGains in
    rebus_synthesis.identification — PrismConfig.from_rebus_synthesis
    duck-types these attributes, so the wire-up exercises identically.

    Values chosen so the default heuristic mapping (0.5/p, 0.5/q,
    0.5*delta_safe) yields plausible gammas in the same order of magnitude
    as the PrismConfig defaults (0.20, 0.15, 0.10) but distinct from them
    (so the differs-from-defaults assertion actually verifies the wire-up).
    """
    p: float = 2.0          # -> gamma_s = 0.25
    q: float = 2.5          # -> gamma_d = 0.20
    delta_safe: float = 0.30  # -> gamma_eps = 0.15
    Gamma: float = 0.50


def synthesize_or_stub() -> tuple[object, str]:
    """Produce a SupervisorGains-shaped object, with provenance label.

    Returns (gains, source) where source is one of:
      'pipeline' — full synthesis ran (cvxpy installed)
      'stub'     — cvxpy missing; using StubSupervisorGains
    """
    try:
        from rebus_synthesis import run_demo, cp
    except ImportError:
        print(
            "rebus_synthesis package not importable; "
            "using StubSupervisorGains to exercise the wire-up."
        )
        return StubSupervisorGains(), "stub"
    if cp is None:
        print(
            "rebus_synthesis present but cvxpy not installed; using "
            "StubSupervisorGains to exercise the wire-up."
        )
        return StubSupervisorGains(), "stub"
    print("running REBUS synthesis pipeline (small synthetic scaffold)...")
    result = run_demo(T=40, nx=2, seed=5, B=6, block_len=8, eta=0.25, solver="SCS")
    return result["gains"], "pipeline"


def main() -> int:
    gains, source = synthesize_or_stub()
    print(
        f"[{source}] gains: p={gains.p:.4f} q={gains.q:.4f} "
        f"delta_safe={gains.delta_safe:.4f} Gamma={gains.Gamma:.4f}"
    )

    cfg = PrismConfig.from_rebus_synthesis(gains)
    print(
        f"PrismConfig.from_rebus_synthesis -> "
        f"gamma_s={cfg.gamma_s:.4f} gamma_d={cfg.gamma_d:.4f} "
        f"gamma_eps={cfg.gamma_eps:.4f}"
    )

    # Sanity: synthesized gammas must differ from defaults — otherwise the
    # wire-up isn't actually driving the config.
    defaults = PrismConfig()
    assert cfg.gamma_s != defaults.gamma_s, "gamma_s did not change from default"
    assert cfg.gamma_d != defaults.gamma_d, "gamma_d did not change from default"
    assert cfg.gamma_eps != defaults.gamma_eps, "gamma_eps did not change from default"

    # Gamma propagation: SupervisorGains.Gamma must flow into cbf_robust_gamma
    # so LATCH's joint-CBF safe set actually shrinks under the synthesizer's
    # disturbance-gain bound. Guards against regression to the pre-PR state
    # where Gamma was dropped on the floor.
    assert cfg.cbf_robust_gamma == gains.Gamma, (
        f"Gamma not propagated: cfg.cbf_robust_gamma={cfg.cbf_robust_gamma} "
        f"vs gains.Gamma={gains.Gamma}"
    )
    assert cfg.cbf_robust_gamma != defaults.cbf_robust_gamma, (
        "cbf_robust_gamma still at default; Gamma wire-up regressed"
    )

    # Custom mapping: prove the gamma_map override works.
    cfg_custom = PrismConfig.from_rebus_synthesis(
        gains,
        gamma_map=lambda g: (0.10, 0.10, 0.10),
    )
    assert cfg_custom.gamma_s == 0.10
    print("custom gamma_map override -> gamma_s=gamma_d=gamma_eps=0.10 (ok)")

    # Overrides kwarg: prove it sets non-gamma fields.
    cfg_override = PrismConfig.from_rebus_synthesis(gains, dt=0.5, seed=42)
    assert cfg_override.dt == 0.5 and cfg_override.seed == 42
    print("**overrides kwarg -> dt=0.5 seed=42 (ok)")

    # Build a model from the synthesized config and run a few forward steps.
    print()
    print("running 10 forward steps with synthesized config...")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = PrismHCLite(cfg)
    state = model.init_state(batch=1)
    belief = model.init_belief(batch=1)
    tele = TelemetryRecorder()

    for t in range(10):
        x = 0.05 * torch.randn(1, cfg.d_in, generator=gen)
        _y, state, belief, rec = model.forward(x, state, belief)
        tele.append_step(rec)

    # All-finite invariant check (rejects both NaN and +/-inf) — proves the
    # synthesized gammas don't blow up the REBUS update.
    for step_idx, r in enumerate(tele.steps):
        for name, v in (("R", r.R), ("E", r.E), ("S", r.S), ("h", r.h),
                        ("F", r.F), ("cbf", r.cbf)):
            assert math.isfinite(v), (
                f"{name} not finite at step {step_idx}: {v!r}"
            )
    print(f"all 10 forward steps finite, F_final={tele.steps[-1].F:.4f}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
