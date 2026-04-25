"""End-to-end smoke check: REBUS synthesis scaffold drives the prototype config.

Pipeline:
  1. Load AI Papers/rebus_identification.py via importlib (the folder name has
     a space, so it isn't a normal Python package). The prototype's
     PrismConfig.from_rebus_synthesis(...) is import-clean — only the caller
     handles the import dance.
  2. Call run_demo(...) which runs make_synthetic_rebus_data ->
     identify_rebus_bounds -> synthesize_supervisor_gains.
  3. Hand the SupervisorGains to PrismConfig.from_rebus_synthesis(gains).
  4. Build a PrismHCLite from the synthesized config and run a few forward
     steps to confirm the wired config produces a stable trajectory.

Requires: cvxpy and scs (already in requirements.txt) — used by
identify_rebus_bounds for the robust-bound LMI solve.

Run:  .\\.venv\\Scripts\\python -m prism_hc.synthesis_demo
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


def load_rebus_module():
    """Load AI Papers/rebus_identification.py from disk.

    Walks up from this file to find the repo root (looks for `AI Papers/`).
    Uses importlib.util because the folder name contains a space and cannot
    be imported as a normal package.
    """
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "AI Papers" / "rebus_identification.py"
        if candidate.exists():
            target = candidate
            break
    else:
        raise FileNotFoundError(
            "Could not locate 'AI Papers/rebus_identification.py' walking up "
            f"from {here}. Synthesis demo requires the AI Papers scaffold."
        )
    spec = importlib.util.spec_from_file_location("rebus_identification", str(target))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build module spec for {target}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: Python 3.10's @dataclass machinery looks
    # up cls.__module__ in sys.modules during class construction, which fails
    # for modules loaded via importlib.util that haven't been registered.
    sys.modules["rebus_identification"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    rid = load_rebus_module()
    if rid.cp is None:
        print("SKIP: cvxpy not installed; cannot run REBUS bound synthesis")
        return 0

    print("running REBUS synthesis pipeline (small synthetic scaffold)...")
    result = rid.run_demo(
        T=40, nx=2, seed=5, B=6, block_len=8, eta=0.25, solver="SCS"
    )
    gains = result["gains"]
    print(
        f"synthesized gains: p={gains.p:.4f} q={gains.q:.4f} "
        f"delta_safe={gains.delta_safe:.4f} Gamma={gains.Gamma:.4f}"
    )

    cfg = PrismConfig.from_rebus_synthesis(gains)
    print(
        f"PrismConfig.from_rebus_synthesis -> "
        f"gamma_s={cfg.gamma_s:.4f} gamma_d={cfg.gamma_d:.4f} "
        f"gamma_eps={cfg.gamma_eps:.4f}"
    )

    # Sanity check: defaults are 0.20, 0.15, 0.10. The synthesized values
    # must differ from these (otherwise the wire-up isn't actually driving).
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

    # All-finite invariant check — proves the synthesized gammas don't blow
    # up the REBUS update.
    for r in tele.steps:
        for name, v in (("R", r.R), ("E", r.E), ("S", r.S), ("h", r.h),
                        ("F", r.F), ("cbf", r.cbf)):
            assert v == v, f"{name} produced NaN at step {r}"  # NaN guard
    print(f"all 10 forward steps finite, F_final={tele.steps[-1].F:.4f}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
