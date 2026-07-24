from __future__ import annotations

import unittest

import torch

from scripts.data.prepare_medical_data import build_reference_mixture_records
from utils.low_rank_fisher_builder import apply_reference_fisher_weights


class ReferenceMixtureTest(unittest.TestCase):
    def test_mixture_records_use_equal_domain_weight(self) -> None:
        records_by_dataset = {
            "bbq": [{"id": f"b{idx}", "messages": [{"role": "assistant", "content": "A"}]} for idx in range(4)],
            "truthfulqa": [{"id": f"t{idx}", "messages": [{"role": "assistant", "content": "yes"}]} for idx in range(4)],
            "mmlu_non_medical": [{"id": f"k{idx}", "messages": [{"role": "assistant", "content": "B"}]} for idx in range(4)],
        }

        records, counts = build_reference_mixture_records(
            records_by_dataset=records_by_dataset,
            mixture_name="B_T_K",
            domains=["bbq", "truthfulqa", "mmlu_non_medical"],
            total_budget=8,
            seed=7,
        )

        self.assertEqual(len(records), 8)
        self.assertEqual(counts, {"bbq": 3, "truthfulqa": 3, "mmlu_non_medical": 2})
        weights_by_domain: dict[str, float] = {}
        for record in records:
            weights_by_domain.setdefault(record["reference_fisher_domain"], 0.0)
            weights_by_domain[record["reference_fisher_domain"]] += float(record["reference_fisher_weight"])
            self.assertEqual(record["reference_mixture"], "B_T_K")
        for total in weights_by_domain.values():
            self.assertAlmostEqual(total, 1.0 / 3.0)
        self.assertAlmostEqual(sum(weights_by_domain.values()), 1.0)

    def test_weighted_rows_match_weighted_outer_product(self) -> None:
        rows = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ]
        )
        weights = torch.tensor([0.25, 0.25, 0.5])

        scaled = apply_reference_fisher_weights(rows, weights)
        actual = scaled.T @ scaled / float(rows.shape[0])
        expected = sum(weights[index] * torch.outer(rows[index], rows[index]) for index in range(rows.shape[0]))

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
