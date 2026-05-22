from __future__ import annotations

import re
from collections import Counter
from typing import Any

import torch

from .common import (
    concat_messages_for_generation,
    generate_response_text,
    get_assistant_target,
    get_prompt_messages,
)


def _normalize_for_overlap(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_for_overlap(text)
    return normalized.split() if normalized else []


def _exact_match(prediction: str, gold: str) -> float:
    return float(_normalize_for_overlap(prediction) == _normalize_for_overlap(gold))


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0

    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(xs: list[str], ys: list[str]) -> int:
    if not xs or not ys:
        return 0

    dp = [0] * (len(ys) + 1)
    for x_token in xs:
        prev = 0
        for col, y_token in enumerate(ys, start=1):
            cached = dp[col]
            if x_token == y_token:
                dp[col] = prev + 1
            else:
                dp[col] = max(dp[col], dp[col - 1])
            prev = cached
    return dp[-1]


def _rouge_l_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 128,
    add_bos_token: bool = False,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    if not subset:
        return {
            "esconv_accuracy": 0.0,
            "esconv_exact_match": 0.0,
            "esconv_token_f1": 0.0,
            "esconv_rouge_l_f1": 0.0,
            "esconv_count": 0.0,
        }

    total = 0
    exact_match_sum = 0.0
    token_f1_sum = 0.0
    rouge_l_f1_sum = 0.0
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
        exact_match = _exact_match(prediction, gold)
        exact_match_sum += exact_match
        token_f1_sum += _token_f1(prediction, gold)
        rouge_l_f1_sum += _rouge_l_f1(prediction, gold)
        total += 1

    exact_match_avg = exact_match_sum / total
    return {
        "esconv_accuracy": float(exact_match_avg),
        "esconv_exact_match": float(exact_match_avg),
        "esconv_token_f1": float(token_f1_sum / total),
        "esconv_rouge_l_f1": float(rouge_l_f1_sum / total),
        "esconv_count": float(total),
    }
