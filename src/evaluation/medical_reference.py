from __future__ import annotations

import re
from collections import Counter
from typing import Any

import torch

from .common import (
    concat_messages_for_generation,
    extract_choice_label,
    generate_response_text,
    get_assistant_target,
    get_prompt_messages,
    normalize_text,
)


def normalize_freeform(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^0-9a-z\s]+", " ", text)
    return " ".join(text.split())


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", normalize_freeform(text), flags=re.UNICODE)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = simple_tokens(prediction)
    gold_tokens = simple_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(gold_tokens))
    return 2.0 * precision * recall / (precision + recall)


def dataset_key(record: dict[str, Any]) -> str:
    source = str(record.get("source", "")).strip().lower()
    task = str(record.get("task", "")).strip().lower()
    if source in {"truthfulqa", "truthful_qa"} or task == "truthfulness":
        return "truthfulqa"
    if source == "bbq" or task == "bias_qa":
        return "bbq"
    return source or task or "unknown"


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 64,
    add_bos_token: bool = False,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    if not subset:
        return {
            "medical_reference_count": 0.0,
            "medical_reference_supported_count": 0.0,
            "medical_reference_unsupported_count": 0.0,
            "bbq_accuracy": 0.0,
            "bbq_count": 0.0,
            "truthfulqa_exact_match": 0.0,
            "truthfulqa_token_f1": 0.0,
            "truthfulqa_count": 0.0,
            "medical_reference_primary_score": 0.0,
        }

    bbq_correct = 0
    bbq_total = 0
    truthfulqa_exact = 0
    truthfulqa_f1_sum = 0.0
    truthfulqa_total = 0
    unsupported_total = 0
    model.eval()

    for record in subset:
        key = dataset_key(record)
        if key not in {"bbq", "truthfulqa"}:
            unsupported_total += 1
            continue
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
        if key == "bbq":
            predicted_label = extract_choice_label(prediction)
            gold_label = extract_choice_label(gold)
            if predicted_label and predicted_label == gold_label:
                bbq_correct += 1
            bbq_total += 1
        elif key == "truthfulqa":
            exact = normalize_freeform(prediction) == normalize_freeform(gold)
            truthfulqa_exact += int(exact)
            truthfulqa_f1_sum += token_f1(prediction, gold)
            truthfulqa_total += 1

    available_scores: list[float] = []
    bbq_accuracy = float(bbq_correct / bbq_total) if bbq_total else 0.0
    truthfulqa_exact_match = float(truthfulqa_exact / truthfulqa_total) if truthfulqa_total else 0.0
    truthfulqa_token_f1 = float(truthfulqa_f1_sum / truthfulqa_total) if truthfulqa_total else 0.0
    if bbq_total:
        available_scores.append(bbq_accuracy)
    if truthfulqa_total:
        available_scores.append(truthfulqa_token_f1)

    supported_total = bbq_total + truthfulqa_total
    return {
        "medical_reference_count": float(len(subset)),
        "medical_reference_supported_count": float(supported_total),
        "medical_reference_unsupported_count": float(unsupported_total),
        "bbq_accuracy": bbq_accuracy,
        "bbq_count": float(bbq_total),
        "truthfulqa_exact_match": truthfulqa_exact_match,
        "truthfulqa_token_f1": truthfulqa_token_f1,
        "truthfulqa_count": float(truthfulqa_total),
        "medical_reference_primary_score": float(sum(available_scores) / len(available_scores))
        if available_scores
        else 0.0,
    }
