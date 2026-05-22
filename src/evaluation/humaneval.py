from __future__ import annotations

import itertools
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from .humaneval_execution import check_correctness

from .common import (
    concat_messages_for_generation,
    generate_response_texts,
    get_assistant_target,
    get_prompt_messages,
    normalize_text,
)


def estimate_pass_at_k(
    num_samples: int | list[int] | np.ndarray,
    num_correct: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, k_value: int) -> float:
        if n - c < k_value:
            return 1.0
        return 1.0 - np.prod(1.0 - k_value / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _sanitize_completion(text: str) -> str:
    completion = _strip_code_fence(text)
    stop_markers = ["\nif __name__ ==", "\n# Test", "\nassert "]
    for marker in stop_markers:
        if marker in completion:
            completion = completion.split(marker, 1)[0]
    return completion.rstrip() + "\n"


def _has_executable_tests(record: dict[str, Any]) -> bool:
    return bool(record.get("prompt") and record.get("test") and record.get("entry_point"))


def _build_problem(record: dict[str, Any], task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "prompt": str(record.get("prompt", "")),
        "test": str(record.get("test", "")),
        "entry_point": str(record.get("entry_point", "")),
    }


def _evaluate_exact_fallback(record: dict[str, Any], completion: str) -> bool:
    return normalize_text(completion) == normalize_text(get_assistant_target(record))


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 256,
    add_bos_token: bool = False,
    timeout: float = 3.0,
    n_workers: int = 4,
    num_samples: int = 1,
    pass_at_ks: tuple[int, ...] = (1,),
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    if not subset:
        metrics = {"humaneval_count": 0.0, "humaneval_executable_count": 0.0, "humaneval_exact_fallback_count": 0.0}
        for k in sorted({int(k_value) for k_value in pass_at_ks if int(k_value) > 0}):
            metrics[f"humaneval_pass_at_{k}"] = 0.0
        return metrics

    task_ids: list[str] = []
    completions_by_task: dict[str, list[str]] = {}
    executable_records: list[tuple[str, dict[str, Any], int, str]] = []
    exact_fallback_count = 0
    executable_count = 0
    resolved_num_samples = max(1, int(num_samples))
    resolved_pass_at_ks = sorted({int(k_value) for k_value in pass_at_ks if int(k_value) > 0})
    if not resolved_pass_at_ks:
        resolved_pass_at_ks = [1]

    for idx, record in enumerate(subset):
        prompt_text = concat_messages_for_generation(
            get_prompt_messages(record),
            tokenizer=tokenizer,
            add_bos_token=add_bos_token,
        )
        predictions = generate_response_texts(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=resolved_num_samples > 1,
            num_return_sequences=resolved_num_samples,
            temperature=temperature,
            top_p=top_p,
        )
        task_id = str(record.get("task_id") or record.get("id") or f"task_{idx}")
        task_ids.append(task_id)
        completions = [_sanitize_completion(prediction) for prediction in predictions]
        completions_by_task[task_id] = completions
        if _has_executable_tests(record):
            executable_count += 1
            for completion_id, completion in enumerate(completions):
                executable_records.append((task_id, record, completion_id, completion))
        else:
            exact_fallback_count += 1

    results: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    if executable_records:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for task_id, record, completion_id, completion in executable_records:
                problem = _build_problem(record, task_id=task_id)
                future = executor.submit(check_correctness, problem, completion, timeout, completion_id)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

    correct_counts_by_task: dict[str, int] = {}
    for task_id in task_ids:
        task_results = results.get(task_id, [])
        task_results.sort()
        correct_counts_by_task[task_id] = sum(1 for _, result in task_results if bool(result["passed"]))

    for task_id, record in zip(task_ids, subset, strict=True):
        task_results = results.get(task_id, [])
        if task_results or _has_executable_tests(record):
            continue
        correct_counts_by_task[task_id] = sum(
            1 for completion in completions_by_task.get(task_id, []) if _evaluate_exact_fallback(record, completion)
        )

    total_arr = np.array([len(completions_by_task[task_id]) for task_id in task_ids], dtype=np.int64)
    correct_arr = np.array([correct_counts_by_task.get(task_id, 0) for task_id in task_ids], dtype=np.int64)

    metrics: dict[str, float] = {
        "humaneval_count": float(len(task_ids)),
        "humaneval_executable_count": float(executable_count),
        "humaneval_exact_fallback_count": float(exact_fallback_count),
    }
    for k in resolved_pass_at_ks:
        pass_at_k = estimate_pass_at_k(total_arr, correct_arr, k).mean() if len(total_arr) > 0 else 0.0
        metrics[f"humaneval_pass_at_{k}"] = float(pass_at_k)
    return metrics
