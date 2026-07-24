from __future__ import annotations

import importlib.util
import unittest

import torch

from safe_drift.safe_drift_selector import (
    _whiten_vectors,
    select_safe_subset_from_gradients,
)


class SafeDriftSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        # With k=3 and eta=1 these gradients produce update atoms
        # [(0.4, 0), (0.2, 0), (0, 0.4), (0, 0.1)].
        self.candidate_gradients = torch.tensor(
            [
                [-1.2, 0.0],
                [-0.6, 0.0],
                [0.0, -1.2],
                [0.0, -0.3],
            ],
            dtype=torch.float32,
        )
        self.candidates = [{"id": index} for index in range(4)]
        self.target_gradient = torch.tensor([-0.5, 0.0], dtype=torch.float32)
        self.fisher = torch.ones(2, dtype=torch.float32)

    def select(self, solver: str):
        return select_safe_subset_from_gradients(
            candidate_gradients=self.candidate_gradients,
            candidates=self.candidates,
            target_gradient=self.target_gradient,
            fisher=self.fisher,
            subset_budget=3,
            alpha=0.0,
            learning_rate=1.0,
            cost_c=0.13,
            epsilon=1.0,
            geometry="reference",
            solver=solver,
            average_by_budget=True,
            greedy_swap_passes=1,
        )

    def test_reference_geometry_includes_alpha_identity_shift(self) -> None:
        vector = torch.tensor([1.0, 2.0])
        fisher = torch.tensor([4.0, 0.0])
        whitened = _whiten_vectors(
            vector,
            fisher,
            geometry="reference",
            alpha=3.0,
        )
        self.assertAlmostEqual(float(torch.dot(whitened, whitened)), 19.0, places=5)

    def test_constrained_greedy_returns_feasible_at_most_budget_subset(self) -> None:
        result = self.select("constrained_greedy")

        self.assertGreater(len(result.selected_indices), 0)
        self.assertLess(len(result.selected_indices), result.subset_budget)
        self.assertLessEqual(result.reference_cost, 0.13 + 1.0e-7)
        self.assertLessEqual(result.norm_cost, 0.5 + 1.0e-7)
        self.assertTrue(all(weight == 1.0 for weight in result.selection_weights))
        self.assertTrue(result.optimization_trace)
        self.assertTrue(
            all(row["reference_budget_satisfied"] for row in result.optimization_trace)
        )
        self.assertTrue(all(row["norm_budget_satisfied"] for row in result.optimization_trace))
        self.assertTrue(
            all(row.get("approximation_error") is not None for row in result.optimization_trace)
        )
        self.assertIn(
            result.solver_status,
            {"no_improving_feasible_addition", "no_feasible_addition"},
        )

    def test_legacy_greedy_records_every_selected_prefix(self) -> None:
        result = self.select("greedy_marginal")

        self.assertEqual(
            len(result.optimization_trace),
            len(result.selected_indices) + 1,
        )
        self.assertEqual(result.optimization_trace[0]["action"], "start")
        self.assertTrue(
            all(
                row.get("approximation_error") is not None
                for row in result.optimization_trace
            )
        )
        self.assertTrue(
            all(
                row.get("reference_budget_satisfied") is not None
                for row in result.optimization_trace
            )
        )

    @unittest.skipUnless(importlib.util.find_spec("cvxpy"), "cvxpy is not installed")
    def test_relaxation_is_feasible_and_no_worse_than_greedy(self) -> None:
        greedy = self.select("constrained_greedy")
        relaxed = self.select("relaxed")

        self.assertTrue(relaxed.selected_indices)
        self.assertTrue(all(0.0 < weight <= 1.0 for weight in relaxed.selection_weights))
        self.assertLessEqual(sum(relaxed.selection_weights), relaxed.subset_budget + 1.0e-6)
        self.assertLessEqual(relaxed.reference_cost, 0.13 + 1.0e-6)
        self.assertLessEqual(relaxed.norm_cost, 0.5 + 1.0e-6)
        self.assertLessEqual(relaxed.objective_value, greedy.objective_value + 1.0e-5)


if __name__ == "__main__":
    unittest.main()
