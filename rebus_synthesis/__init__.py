"""REBUS identification + robust supervisor-gain synthesis.

Public surface re-exported from `.identification` so callers can write
`from rebus_synthesis import run_demo, SupervisorGains, cp` without
reaching into submodules. The optional `cp` sentinel is `None` when
cvxpy is not installed; consumers gate solver-dependent paths on it.

Promoted from `AI Papers/rebus_identification.py` so PRISM-HC's
synthesis_demo.py can import the scaffold normally instead of via
importlib.util.spec_from_file_location ancestor-walking.
"""

from .identification import (
    RebusBounds,
    SupervisorGains,
    SyntheticTruth,
    cp,
    identify_rebus_bounds,
    make_synthetic_rebus_data,
    run_demo,
    synthesize_supervisor_gains,
)

__all__ = [
    "RebusBounds",
    "SupervisorGains",
    "SyntheticTruth",
    "cp",
    "identify_rebus_bounds",
    "make_synthetic_rebus_data",
    "run_demo",
    "synthesize_supervisor_gains",
]
