from __future__ import annotations

import math
from typing import Any


def planned_optimizer_step_count(args: Any, subset_budget: int) -> int:
    if args.max_steps is not None:
        return max(1, int(args.max_steps))
    global_batch_size = (
        int(args.per_device_train_batch_size)
        * int(args.gradient_accumulation_steps)
        * int(args.num_processes)
    )
    if global_batch_size <= 0:
        raise ValueError("The effective training batch size must be positive.")
    update_steps_per_epoch = max(1, math.ceil(int(subset_budget) / global_batch_size))
    return max(1, math.ceil(float(args.num_train_epochs) * update_steps_per_epoch))


def _schedule_factor(
    scheduler_type: str,
    *,
    step: int,
    warmup_steps: int,
    training_steps: int,
    base_learning_rate: float,
) -> float:
    scheduler = str(scheduler_type).strip().lower()
    if scheduler == "constant":
        return 1.0
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if scheduler == "constant_with_warmup":
        return 1.0

    decay_steps = max(1, training_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler == "linear":
        return max(0.0, 1.0 - progress)
    if scheduler == "cosine":
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if scheduler == "cosine_with_restarts":
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress * 1.0) % 1.0))))
    if scheduler == "polynomial":
        end_learning_rate = min(1.0e-7, float(base_learning_rate))
        decayed = (
            (float(base_learning_rate) - end_learning_rate) * (1.0 - progress)
            + end_learning_rate
        )
        return decayed / float(base_learning_rate)
    raise ValueError(
        f"Unsupported scheduler for automatic Adam selection scaling: {scheduler_type!r}. "
        "Set --adam-selection-effective-learning-rate explicitly."
    )


def planned_learning_rate_sum(args: Any, subset_budget: int) -> tuple[float, dict[str, Any]]:
    max_steps = planned_optimizer_step_count(args, subset_budget)
    warmup_steps = int(math.ceil(max_steps * float(args.warmup_ratio)))
    base_learning_rate = float(args.learning_rate)
    if base_learning_rate <= 0.0:
        raise ValueError("The training learning rate must be positive.")

    learning_rate_sum = base_learning_rate * sum(
        _schedule_factor(
            args.lr_scheduler_type,
            step=step,
            warmup_steps=warmup_steps,
            training_steps=max_steps,
            base_learning_rate=base_learning_rate,
        )
        for step in range(max_steps)
    )
    return float(learning_rate_sum), {
        "adam_selection_training_step_count": max_steps,
        "adam_selection_warmup_step_count": warmup_steps,
        "adam_selection_base_learning_rate": base_learning_rate,
        "adam_selection_lr_scheduler_type": str(args.lr_scheduler_type),
        "adam_selection_effective_batch_size": (
            int(args.per_device_train_batch_size)
            * int(args.gradient_accumulation_steps)
            * int(args.num_processes)
        ),
    }
