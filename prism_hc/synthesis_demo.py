"""End-to-end smoke check: REBUS synthesis scaffold drives the prototype config.

The scaffold under `AI Papers/rebus_identification.py` is currently
out-of-tree (not committed to this repo). This demo handles three cases:

  1. Scaffold present + cvxpy available: run the full synthesis pipeline
     (make_synthetic_rebus_data -> identify_rebus_bounds ->
     synthesize_supervisor_gains) and feed the gains into PrismConfig.
  2. Scaffold present but cvxpy missing: skip the LMI solve and exercise
     the wire-up with a fabricated SupervisorGains-shaped object.
  3. Scaffold absent (default for fresh clones): same fallback as (2).

Cases (2) and (3) prove that PrismConfig.from_rebus_synthesis is
import-clean — it duck-types its `gains` argument and never imports
from `AI Papers/`, so the prototype's wire-up surface is testable
without the out-of-tree scaffold.

Run:  python -m prism_hc.synthesis_demo
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


@dataclass(frozen=True)
class StubSupervisorGains:
    """Stand-in for SupervisorGains when the AI Papers scaffold isn't present.

    Field names and semantics match SupervisorGains at
    AI Papers/rebus_identification.py:75 — PrismConfig.from_rebus_synthesis
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


def find_rebus_module_path() -> Optional[Path]:
    """Walk up from this file looking for AI Papers/rebus_identification.py."""
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "AI Papers" / "rebus_identification.py"
        if candidate.exists():
            return candidate
    return None


def load_rebus_module(target: Path):
    """Load rebus_identification.py from `target` via importlib.

    The folder name has a space, so it can't be imported as a package.
    """
    spec = importlib.util.spec_from_file_location("rebus_identification", str(target))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build module spec for {target}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: Python 3.10's @dataclass machinery looks
    # up cls.__module__ in sys.modules during class construction, which
    # fails for modules loaded via importlib that haven't been registered.
    sys.modules["rebus_identification"] = module
    spec.loader.exec_module(module)
    return module


def synthesize_or_stub() -> tuple[object, str]:
    """Produce a SupervisorGains-shaped object, with provenance label.

    Returns (gains, source) where source is one of:
      'pipeline' — full synthesis ran (scaffold + cvxpy)
      'stub'     — scaffold absent or cvxpy missing; using StubSupervisorGains
    """
    target = find_rebus_module_path()
    if target is None:
        print(
            "AI Papers/rebus_identification.py not found in the repo tree; "
            "using StubSupervisorGains to exercise the wire-up."
        )
        return StubSupervisorGains(), "stub"
    rid = load_rebus_module(target)
    if rid.cp is None:
        print(
            "scaffold present but cvxpy not installed; using "
            "StubSupervisorGains to exercise the wire-up."
        )
        return StubSupervisorGains(), "stub"
    print("running REBUS synthesis pipeline (small synthetic scaffold)...")
    result = rid.run_demo(
        T=40, nx=2, seed=5, B=6, block_len=8, eta=0.25, solver="SCS"
    )
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
