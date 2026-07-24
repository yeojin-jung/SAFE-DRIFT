from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from utils.adam_selection import planned_learning_rate_sum
from utils.low_rank_fisher_builder import build_low_rank_safe_inputs_from_gradient_rows


class AdamSelectionTest(unittest.TestCase):
    def test_preconditioning_before_projection_keeps_reentering_components(self) -> None:
        reference_rows = torch.tensor([[1.0, 1.0]])
        target_gradient = torch.tensor([1.0, 1.0])
        candidate_rows = torch.tensor([[1.0, -1.0]])
        preconditioner = torch.tensor([1.0, 3.0])

        legacy_inputs = build_low_rank_safe_inputs_from_gradient_rows(
            reference_grad_rows=reference_rows,
            g_T=target_gradient,
            candidate_grad_rows=candidate_rows,
            alpha=1.0e-3,
            K_R=1,
            K_T=0,
            K_common=1,
        )
        basis = legacy_inputs.common_basis.U_K
        projected_preconditioner = basis.T @ (preconditioner[:, None] * basis)
        legacy = legacy_inputs.candidate_features @ projected_preconditioner.T

        corrected_inputs = build_low_rank_safe_inputs_from_gradient_rows(
            reference_grad_rows=reference_rows,
            g_T=target_gradient,
            candidate_grad_rows=candidate_rows * preconditioner,
            task_basis_gradient=target_gradient * preconditioner,
            alpha=1.0e-3,
            K_R=1,
            K_T=0,
            K_common=1,
        )
        expected = (candidate_rows * preconditioner) @ basis

        self.assertTrue(torch.allclose(legacy, torch.zeros_like(legacy), atol=1.0e-6))
        self.assertTrue(
            torch.allclose(corrected_inputs.candidate_features, expected, atol=1.0e-6)
        )
        self.assertGreater(abs(float(corrected_inputs.candidate_features[0, 0])), 1.0)

    def test_task_basis_can_be_built_from_update_directions(self) -> None:
        reference_rows = torch.tensor([[1.0, 0.0, 0.0]])
        target_gradient = torch.tensor([0.0, 10.0, 0.0])
        candidate_rows = torch.tensor([[0.0, 0.0, 1.0]])
        preconditioner = torch.tensor([1.0, 0.01, 1.0])

        legacy = build_low_rank_safe_inputs_from_gradient_rows(
            reference_grad_rows=reference_rows,
            g_T=target_gradient,
            candidate_grad_rows=candidate_rows,
            alpha=1.0e-3,
            K_R=1,
            K_T=1,
            K_common=2,
        )
        corrected = build_low_rank_safe_inputs_from_gradient_rows(
            reference_grad_rows=reference_rows,
            g_T=target_gradient,
            candidate_grad_rows=candidate_rows * preconditioner,
            task_basis_gradient=target_gradient * preconditioner,
            alpha=1.0e-3,
            K_R=1,
            K_T=1,
            K_common=2,
        )

        legacy_task_direction = legacy.common_basis.U_add[:, 0]
        corrected_task_direction = corrected.common_basis.U_add[:, 0]
        self.assertGreater(abs(float(legacy_task_direction[1])), 0.99)
        self.assertGreater(abs(float(corrected_task_direction[2])), 0.99)

    def test_planned_learning_rate_sum_matches_constant_schedule(self) -> None:
        args = SimpleNamespace(
            max_steps=None,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_processes=1,
            num_train_epochs=1.0,
            warmup_ratio=0.0,
            learning_rate=0.1,
            lr_scheduler_type="constant",
        )

        value, metadata = planned_learning_rate_sum(args, subset_budget=10)

        self.assertAlmostEqual(value, 0.3, places=7)
        self.assertEqual(metadata["adam_selection_training_step_count"], 3)
        self.assertEqual(metadata["adam_selection_effective_batch_size"], 4)

    def test_planned_learning_rate_sum_replays_linear_warmup_and_decay(self) -> None:
        args = SimpleNamespace(
            max_steps=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_processes=1,
            num_train_epochs=1.0,
            warmup_ratio=0.25,
            learning_rate=0.1,
            lr_scheduler_type="linear",
        )

        value, metadata = planned_learning_rate_sum(args, subset_budget=100)

        self.assertAlmostEqual(value, 0.2, places=7)
        self.assertEqual(metadata["adam_selection_training_step_count"], 4)
        self.assertEqual(metadata["adam_selection_warmup_step_count"], 1)


if __name__ == "__main__":
    unittest.main()
