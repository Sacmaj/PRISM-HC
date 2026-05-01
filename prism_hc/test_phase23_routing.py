"""Phase 2/3 routing, interface, and decoder-gate tests."""

from __future__ import annotations

import unittest

import torch

from prism_hc.config import PrismConfig
from prism_hc.interfaces import PlasticityController, Router, SafetyAnchorCore
from prism_hc.model import PrismHCLite
from prism_hc.state import SafetyState


class InterfaceProtocolTests(unittest.TestCase):
    def test_runtime_protocols_match_current_modules(self) -> None:
        model = PrismHCLite(PrismConfig())
        self.assertIsInstance(model.latch, PlasticityController)
        self.assertIsInstance(model.anchors, SafetyAnchorCore)
        self.assertIsInstance(model.reservoir, Router)


class SparseRoutingTests(unittest.TestCase):
    def _model_state_belief(self, **overrides):
        cfg = PrismConfig(**overrides)
        model = PrismHCLite(cfg)
        state = model.init_state(batch=2)
        belief = model.init_belief(batch=2)
        return cfg, model, state, belief

    def test_forward_records_top_k_without_health_update(self) -> None:
        cfg, model, state, belief = self._model_state_belief(route_top_k=3)
        health_before = state.reservoir.route_health.clone()
        hits_before = state.reservoir.hits.clone()
        x = torch.randn(2, cfg.d_in)

        _y, state, _belief, rec = model.forward(x, state, belief)

        self.assertTrue(torch.equal(state.reservoir.route_health, health_before))
        self.assertTrue(torch.equal(state.reservoir.hits, hits_before))
        active_per_sample = state.reservoir.active_paths.sum(dim=-1)
        self.assertTrue(torch.all(active_per_sample <= cfg.route_top_k))
        self.assertLessEqual(rec.active_paths, cfg.route_top_k)
        self.assertGreaterEqual(rec.route_entropy, 0.0)

    def test_default_dense_routing_ignores_health_prune_threshold(self) -> None:
        cfg, model, state, belief = self._model_state_belief(
            route_top_k=None,
            route_prune_threshold=1.0,
        )
        state.reservoir.route_health.zero_()
        x = torch.randn(2, cfg.d_in)

        _y, state, _belief, rec = model.forward(x, state, belief)

        self.assertTrue(torch.all(state.reservoir.active_paths))
        self.assertEqual(rec.active_paths, cfg.d_reservoir)

    def test_post_score_update_mutates_health_and_hits(self) -> None:
        cfg, model, state, belief = self._model_state_belief(route_top_k=3)
        x = torch.randn(2, cfg.d_in)
        _y, state, _belief, _rec = model.forward(x, state, belief)
        health_before = state.reservoir.route_health.clone()
        hits_before = state.reservoir.hits.clone()

        state = model.post_score_update(state, reward=torch.tensor(1.0))

        self.assertFalse(torch.equal(state.reservoir.route_health, health_before))
        self.assertFalse(torch.equal(state.reservoir.hits, hits_before))
        self.assertEqual(int(state.reservoir.hits.sum().item()), 2 * cfg.route_top_k)

    def test_health_decay_prune_keeps_topology_bounded(self) -> None:
        cfg, model, state, belief = self._model_state_belief(
            route_top_k=2,
            route_health_decay=0.5,
            route_health_reward_rate=0.0,
            route_health_floor=0.05,
            route_prune_threshold=0.25,
        )
        x = torch.randn(2, cfg.d_in)

        for _ in range(8):
            _y, state, belief, rec = model.forward(x, state, belief)
            self.assertLessEqual(rec.active_paths, cfg.route_top_k)
            state = model.post_score_update(state, reward=torch.tensor(0.0))

        self.assertGreaterEqual(float(state.reservoir.route_health.min().item()), 0.05)
        self.assertLessEqual(float(state.reservoir.route_health.max().item()), 1.0)
        self.assertGreaterEqual(state.topology.topology_mass, 0.0)
        self.assertLessEqual(state.topology.topology_mass, 1.0)
        self.assertLessEqual(state.topology.prune_budget, cfg.d_reservoir)

    def test_reset_clears_route_and_topology_fast_state(self) -> None:
        cfg, model, state, belief = self._model_state_belief(route_top_k=3)
        U_snapshot = model.anchors.U.clone()
        W_snapshot = model.reservoir.W_rand.clone()
        x = torch.randn(2, cfg.d_in)

        _y, state, _belief, _rec = model.forward(x, state, belief)
        state = model.post_score_update(state, reward=torch.tensor(1.0))
        self.assertGreater(float(state.reservoir.hits.sum().item()), 0.0)

        state = model.reset_episode(state)

        self.assertTrue(torch.equal(model.anchors.U, U_snapshot))
        self.assertTrue(torch.equal(model.reservoir.W_rand, W_snapshot))
        self.assertTrue(torch.equal(
            state.reservoir.route_health,
            torch.ones_like(state.reservoir.route_health),
        ))
        self.assertTrue(torch.equal(
            state.reservoir.hits,
            torch.zeros_like(state.reservoir.hits),
        ))
        self.assertTrue(torch.equal(
            state.reservoir.active_paths,
            torch.zeros_like(state.reservoir.active_paths),
        ))
        self.assertEqual(state.topology.active_edges, 0)
        self.assertEqual(state.topology.topology_mass, 0.0)


class DecoderGateTests(unittest.TestCase):
    def test_decoder_gate_is_noop_by_default(self) -> None:
        model = PrismHCLite(PrismConfig())
        y = torch.randn(2, model.cfg.d_in)
        safety = SafetyState(
            anchor_coords={},
            drift=torch.tensor(1.0),
            canary_margin=0.0,
        )

        gated, vetoes = model.apply_decoder_gate(y, safety)

        self.assertTrue(torch.equal(gated, y))
        self.assertEqual(vetoes, 0)

    def test_decoder_gate_penalizes_configured_indices_under_low_canary(self) -> None:
        cfg = PrismConfig(
            decoder_veto_indices=(0, 2),
            decoder_veto_penalty=3.5,
            decoder_canary_threshold=0.8,
        )
        model = PrismHCLite(cfg)
        y = torch.zeros(2, cfg.d_in)
        low_safety = SafetyState(
            anchor_coords={},
            drift=torch.tensor(0.9),
            canary_margin=0.2,
        )

        gated, vetoes = model.apply_decoder_gate(y, low_safety)

        self.assertEqual(vetoes, 4)
        self.assertTrue(torch.all(gated[:, [0, 2]] == -3.5))
        self.assertTrue(torch.all(gated[:, [1, 3, 4, 5, 6, 7]] == 0.0))

        high_safety = SafetyState(
            anchor_coords={},
            drift=torch.tensor(0.0),
            canary_margin=0.9,
        )
        no_op, vetoes = model.apply_decoder_gate(y, high_safety)
        self.assertTrue(torch.equal(no_op, y))
        self.assertEqual(vetoes, 0)

    def test_decoder_veto_indices_are_deduplicated(self) -> None:
        cfg = PrismConfig(
            decoder_veto_indices=(0, 0, 2),
            decoder_veto_penalty=1.0,
            decoder_canary_threshold=0.8,
        )
        self.assertEqual(cfg.decoder_veto_indices, (0, 2))
        model = PrismHCLite(cfg)
        y = torch.zeros(2, cfg.d_in)
        safety = SafetyState(
            anchor_coords={},
            drift=torch.tensor(0.9),
            canary_margin=0.2,
        )

        gated, vetoes = model.apply_decoder_gate(y, safety)

        self.assertEqual(vetoes, 4)
        self.assertTrue(torch.all(gated[:, [0, 2]] == -1.0))


if __name__ == "__main__":
    unittest.main()
