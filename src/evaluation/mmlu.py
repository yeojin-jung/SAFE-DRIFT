from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .common import (
    concat_messages_for_generation,
    extract_choice_label,
    generate_response_text,
    get_assistant_target,
    get_prompt_messages,
)


CHOICE_LABELS = ("A", "B", "C", "D")


def _choice_label(record: dict[str, Any]) -> str:
    explicit = record.get("answer")
    if isinstance(explicit, str) and explicit.strip().upper() in CHOICE_LABELS:
        return explicit.strip().upper()
    if isinstance(explicit, int) and 0 <= explicit < len(CHOICE_LABELS):
        return CHOICE_LABELS[explicit]
    return extract_choice_label(get_assistant_target(record))


@torch.no_grad()
def _continuation_logprob(
    *,
    model,
    tokenizer,
    prompt_text: str,
    continuation: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    continuation_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if continuation_ids.numel() == 0:
        return float("-inf")

    input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    logits = model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_ids.shape[1] - 1)
    end = start + continuation_ids.shape[1]
    return float(token_log_probs[:, start:end].sum().item())


def _loglikelihood_prediction(
    *,
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
) -> str:
    scores = {
        label: _continuation_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            continuation=label,
            device=device,
        )
        for label in CHOICE_LABELS
    }
    return max(scores, key=scores.get)


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
        return {
            "mmlu_chat_accuracy": 0.0,
            "mmlu_loglikelihood_accuracy": 0.0,
            "mmlu_accuracy": 0.0,
            "mmlu_count": 0.0,
        }

    chat_correct = 0
    loglikelihood_correct = 0
    total = 0
    model.eval()
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
        gold = _choice_label(record)
        chat_prediction = extract_choice_label(prediction)
        loglikelihood_prediction = _loglikelihood_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            device=device,
        )
        if chat_prediction and chat_prediction == gold:
            chat_correct += 1
        if loglikelihood_prediction == gold:
            loglikelihood_correct += 1
        total += 1

    return {
        "mmlu_chat_accuracy": float(chat_correct / total),
        "mmlu_loglikelihood_accuracy": float(loglikelihood_correct / total),
        "mmlu_accuracy": float(chat_correct / total),
        "mmlu_count": float(total),
    }
