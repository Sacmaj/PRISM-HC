"""Per-step and per-commit telemetry records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StepRecord:
    R: float
    E: float
    S: float
    rho: float
    chi: float
    P: float
    h: float
    F: float
    drift: float
    cbf: float
    dwell: int = 0


@dataclass
class CommitRecord:
    committed: bool
    reason: str
    g_norm: float
    step_index: int = -1


@dataclass
class TelemetryRecorder:
    steps: List[StepRecord] = field(default_factory=list)
    commits: List[CommitRecord] = field(default_factory=list)

    def append_step(self, rec: StepRecord) -> None:
        self.steps.append(rec)

    def append_commit(self, rec: CommitRecord) -> None:
        if rec.step_index < 0:
            rec.step_index = len(self.steps) - 1
        self.commits.append(rec)

    def print_table(self) -> None:
        header = (
            f"{'t':>3} {'R':>6} {'E':>6} {'S':>6} {'dwell':>5} "
            f"{'rho':>6} {'P':>6} {'h':>6} {'F':>8} {'drift':>6} {'cbf':>7}"
        )
        print(header)
        print("-" * len(header))
        for t, r in enumerate(self.steps):
            print(
                f"{t:>3d} {r.R:>6.3f} {r.E:>6.3f} {r.S:>6.3f} {r.dwell:>5d} "
                f"{r.rho:>6.3f} {r.P:>6.3f} {r.h:>6.3f} {r.F:>8.3f} "
                f"{r.drift:>6.3f} {r.cbf:>+7.3f}"
            )
        for c in self.commits:
            print(
                f"  commit @ t={c.step_index}: committed={c.committed} "
                f"reason={c.reason} g_norm={c.g_norm:.4f}"
            )
