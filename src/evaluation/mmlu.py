from __future__ import annotations

from typing import Any

import torch

from .common import (
    concat_messages_for_generation,
    extract_choice_label,
    generate_response_text,
    get_assistant_target,
    get_prompt_messages,
)


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 32,
    add_bos_token: bool = False,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    if not subset:
        return {"mmlu_accuracy": 0.0, "mmlu_count": 0.0}

    correct = 0
    total = 0
    for record in subset:
        prompt = concat_messages_for_generation(
            get_prompt_messages(record),
            tokenizer=tokenizer,
            add_bos_token=add_bos_token,
        )
        prediction = generate_response_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gold = get_assistant_target(record)
        if extract_choice_label(prediction) and extract_choice_label(prediction) == extract_choice_label(gold):
            correct += 1
        total += 1

    return {
        "mmlu_accuracy": float(correct / total),
        "mmlu_count": float(total),
    }
