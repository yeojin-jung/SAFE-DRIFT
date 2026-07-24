from __future__ import annotations

from typing import Any

import torch

from . import medqa


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 32,
    add_bos_token: bool = False,
) -> dict[str, float]:
    metrics = medqa.evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        device=device,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
        add_bos_token=add_bos_token,
    )
    return {
        "mcq_ood_accuracy": float(metrics["medqa_accuracy"]),
        "mcq_ood_count": float(metrics["medqa_count"]),
    }
