"""Unit tests for null-space projection used by the plasticity adapter."""

from __future__ import annotations

import unittest

import torch

from prism_hc.adapter import NullSpaceAdapter
from prism_hc.config import PrismConfig
from prism_hc.model import PrismHCLite
from prism_hc.telemetry import TelemetryRecorder


class NullSpaceProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        H, k = 16, 4
        raw = torch.randn(H, k)
        self.U, _ = torch.linalg.qr(raw)
        self.H = H
        self.k = k

    def test_projection_orthogonal_to_U(self) -> None:
        g = torch.randn(8, self.H)
        g_safe = NullSpaceAdapter.project_update(g, self.U)
        residual = g_safe @ self.U
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-5))

    def test_projection_idempotent(self) -> None:
        g = torch.randn(8, self.H)
        g1 = NullSpaceAdapter.project_update(g, self.U)
        g2 = NullSpaceAdapter.project_update(g1, self.U)
        self.assertTrue(torch.allclose(g1, g2, atol=1e-6))

    def test_projection_preserves_null_components(self) -> None:
        # A vector with zero anchor-component must pass through unchanged.
        anchor_part = self.U @ torch.randn(self.k)
        full = torch.randn(self.H)
        null_part = full - self.U @ (self.U.T @ full)
        out = NullSpaceAdapter.project_update(null_part, self.U)
        self.assertTrue(torch.allclose(out, null_part, atol=1e-6))
        # And mixing in an anchor part must be removed.
        mixed = null_part + anchor_part
        out2 = NullSpaceAdapter.project_update(mixed, self.U)
        self.assertTrue(torch.allclose(out2, null_part, atol=1e-5))

    def test_skips_dim_mismatch(self) -> None:
        # Bias-style 1-D tensor with last-dim != H must pass through unchanged.
        g = torch.randn(7)
        out = NullSpaceAdapter.project_update(g, self.U)
        self.assertTrue(torch.equal(out, g))


class AuthorityAsymmetryProbeTests(unittest.TestCase):
    """End-to-end null-space invariant across many commit cycles.

    The standalone projection tests above verify project_update's geometry in
    isolation. This test exercises the full plasticity_step pipeline (gate
    check, projection, SGD apply, priming drain, chi accumulation) for N=100
    forced commits and asserts that every accumulated parameter delta still
    lies in null(U) to within 1e-5, and that the slow state (anchors.U,
    reservoir.W_rand) is bit-identical at the end.
    """

    def test_authority_asymmetry_over_n_commits(self) -> None:
        torch.manual_seed(0)
        cfg = PrismConfig()
        model = PrismHCLite(cfg)
        state = model.init_state(batch=1)
        tele = TelemetryRecorder()
        # Snapshot trainable params and slow state.
        param_snapshot = {n: p.detach().clone()
                          for n, p in model.named_parameters()}
        U_snapshot = model.anchors.U.clone()
        W_snapshot = model.reservoir.W_rand.clone()
        U = model.anchors.U
        gen = torch.Generator().manual_seed(1)
        N = 100
        for _ in range(N):
            # Force the gate open every iteration: P drains by commit_cost
            # per accept, so without refilling the gate closes after ~2 commits.
            state.S.fill_(0.95)
            state.E.fill_(0.10)
            state.rho.fill_(0.10)
            state.dwell_counter.fill_(cfg.dwell_min + 1)
            state.P.fill_(1.0)
            grads = {
                n: torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device) * 0.01
                for n, p in model.named_parameters() if p.requires_grad
            }
            model.plasticity_step(state, grads, tele)
        # For each hidden-aligned param, accumulated delta has zero U-component.
        violations: list = []
        for n, p in model.named_parameters():
            if p.shape[-1] != U.shape[0]:
                # Skip biases / readout (last-dim is d_in, not d_hidden).
                continue
            delta = p.detach() - param_snapshot[n]
            flat = delta.reshape(-1, delta.shape[-1])
            residual = (flat @ U).abs().max().item()
            if residual >= 1e-5:
                violations.append((n, residual))
        self.assertFalse(
            violations,
            f"Params drifted into span(U) over {N} commits: {violations}",
        )
        # Slow state untouched.
        self.assertTrue(torch.equal(model.anchors.U, U_snapshot),
                        "anchors.U mutated during plasticity cycles")
        self.assertTrue(torch.equal(model.reservoir.W_rand, W_snapshot),
                        "reservoir.W_rand mutated during plasticity cycles")
        # Sanity: most commits accepted (guards a silent gate-closure regression).
        accepted = sum(1 for c in tele.commits if c.committed)
        self.assertGreater(accepted, N // 2,
                           f"only {accepted}/{N} commits accepted")


if __name__ == "__main__":
    unittest.main()
