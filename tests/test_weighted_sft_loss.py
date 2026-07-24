from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from utils.train_lora_sft import (
    constraint_audit,
    cumulative_safe_scale,
    largest_scale_for_quadratic,
    safe_training_update_scale,
    weighted_causal_lm_loss,
)


class WeightedSftLossTest(unittest.TestCase):
    def test_constraint_audit_marks_budget_violations(self) -> None:
        satisfied = constraint_audit(cost=0.9, budget=1.0, prefix="test")
        violated = constraint_audit(cost=1.1, budget=1.0, prefix="test")
        self.assertTrue(satisfied["test_budget_satisfied"])
        self.assertFalse(violated["test_budget_satisfied"])
        self.assertAlmostEqual(float(violated["test_budget_ratio"]), 1.1)

    def test_unit_weights_match_standard_token_mean_loss(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(2, 4, 5)
        labels = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, -100, 3, 4],
            ]
        )
        expected = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        actual = weighted_causal_lm_loss(logits, labels, torch.ones(2))
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))

    def test_zero_weight_removes_an_examples_token_losses(self) -> None:
        torch.manual_seed(11)
        logits = torch.randn(2, 3, 4)
        labels = torch.tensor([[0, 1, 2], [1, 2, 3]])
        actual = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0, 0.0]))

        first_losses = F.cross_entropy(
            logits[0, :-1, :],
            labels[0, 1:],
            reduction="sum",
        )
        # The denominator remains the ordinary full-batch valid-token count;
        # this makes weight=1 exactly recover the existing training loss.
        expected = first_losses / 4.0
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))


class SafeTrainingProjectionTest(unittest.TestCase):
    def test_quadratic_solver_returns_largest_feasible_scale(self) -> None:
        scale = largest_scale_for_quadratic(
            quadratic=1.0,
            linear=0.0,
            constant=-0.25,
        )
        self.assertAlmostEqual(scale, 0.5, places=6)

    def test_cumulative_projection_accounts_for_cross_terms(self) -> None:
        scale, diagnostics = cumulative_safe_scale(
            displacement=torch.tensor([0.6]),
            proposal=torch.tensor([0.6]),
            fisher=torch.tensor([1.0]),
            rho=0.5,
            epsilon=1.0,
        )
        self.assertAlmostEqual(scale, 2.0 / 3.0, places=5)
        self.assertLessEqual(float(diagnostics["accepted_cumulative_reference_cost"]), 0.5)
        self.assertLessEqual(float(diagnostics["accepted_cumulative_norm_cost"]), 0.5)

    def test_inward_proposal_is_not_clipped(self) -> None:
        scale, _ = cumulative_safe_scale(
            displacement=torch.tensor([0.8]),
            proposal=torch.tensor([-1.0]),
            fisher=torch.tensor([1.0]),
            rho=0.5,
            epsilon=1.0,
        )
        self.assertEqual(scale, 1.0)

    def test_equal_allocation_uses_per_step_budgets(self) -> None:
        scale, diagnostics = safe_training_update_scale(
            mode="equal_allocation",
            displacement=torch.zeros(1),
            proposal=torch.tensor([1.0]),
            fisher=torch.tensor([1.0]),
            rho=0.5,
            epsilon=1.0,
            max_steps=4,
        )
        self.assertAlmostEqual(scale, 0.25, places=5)
        self.assertAlmostEqual(float(diagnostics["equal_allocation_step_epsilon"]), 0.25)
        self.assertLessEqual(
            float(diagnostics["equal_allocation_accepted_update_reference_cost"]),
            0.5 / 16.0,
        )


if __name__ == "__main__":
    unittest.main()
