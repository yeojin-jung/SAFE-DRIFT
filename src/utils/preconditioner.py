import random

import torch
from tqdm import tqdm

from .extract_gradients import compute_per_example_gradient


def estimate_adam_second_moment(model, tokenizer, warmup_data,
                                beta2, n_steps, device, max_seq_len,
                                use_lora=True):
    if not warmup_data:
        raise ValueError("warmup_data must not be empty.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    model.eval()
    v_bar = None

    for _ in tqdm(range(n_steps), desc="Adam warmup"):
        example = random.choice(warmup_data)
        g = compute_per_example_gradient(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=max_seq_len,
            use_lora=use_lora,
        )

        if v_bar is None:
            v_bar = (1 - beta2) * g.pow(2)
        else:
            v_bar = beta2 * v_bar + (1 - beta2) * g.pow(2)

    correction = 1.0 - beta2 ** n_steps
    return v_bar / correction


def build_adam_preconditioner(v_bar, eps, eta=1.0):
    return eta / (v_bar.sqrt() + eps)


def build_sgd_preconditioner(d_lora, eta=1.0):
    return torch.full((d_lora,), float(eta))
