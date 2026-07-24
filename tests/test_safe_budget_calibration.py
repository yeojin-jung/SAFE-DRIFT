from __future__ import annotations

import unittest

import torch

from utils.safe_budget_calibration import (
    coupled_reference_budget,
    random_subset_update_norm_calibration,
)


class SafeBudgetCalibrationTest(unittest.TestCase):
    def test_random_subset_update_norm_is_seeded_and_uses_preconditioner(self) -> None:
        gradients = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        )
        kwargs = {
            "subset_budget": 2,
            "learning_rate": 0.5,
            "preconditioner": torch.tensor([2.0, 1.0]),
            "sample_count": 7,
            "seed": 13,
        }

        first, first_info = random_subset_update_norm_calibration(gradients, **kwargs)
        second, second_info = random_subset_update_norm_calibration(gradients, **kwargs)

        self.assertAlmostEqual(first, second)
        self.assertEqual(
            first_info["safe_epsilon_calibration_update_norms"],
            second_info["safe_epsilon_calibration_update_norms"],
        )
        self.assertTrue(first_info["safe_epsilon_calibration_uses_preconditioner"])
        self.assertGreater(first, 0.0)

    def test_coupled_reference_budget_matches_gamma_definition(self) -> None:
        rho, info = coupled_reference_budget(
            target_gradient=torch.tensor([3.0, 4.0]),
            reference_fisher=torch.tensor([2.0, 8.0]),
            preconditioner=torch.tensor([2.0, 1.0]),
            epsilon=0.5,
            gamma=0.1,
        )

        direction = torch.tensor([-6.0, -4.0])
        direction = direction / torch.linalg.vector_norm(direction)
        q0 = float(torch.dot(direction, torch.tensor([2.0, 8.0]) * direction))
        self.assertAlmostEqual(info["safe_reference_curvature_q0"], q0, places=6)
        self.assertAlmostEqual(rho, 0.5 * 0.1 * q0 * 0.5**2, places=6)


if __name__ == "__main__":
    unittest.main()
