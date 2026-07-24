from __future__ import annotations

from collections.abc import Callable

from . import bias_disentangle, esconv, gsm8k, humaneval, ifeval, ifeval_proxy, mcq_ood, medical_reference, medqa, mmlu


def _evaluate_none(*_args, **_kwargs) -> dict[str, float]:
    return {}


EVALUATORS: dict[str, Callable[..., dict[str, float]]] = {
    "none": _evaluate_none,
    "medqa": medqa.evaluate_records,
    "mmlu": mmlu.evaluate_records,
    "gsm8k": gsm8k.evaluate_records,
    "esconv": esconv.evaluate_records,
    "humaneval": humaneval.evaluate_records,
    "ifeval": ifeval.evaluate_records,
    "ifeval_proxy": ifeval_proxy.evaluate_records,
    "medical_reference": medical_reference.evaluate_records,
    "mcq_ood": mcq_ood.evaluate_records,
    "bias_disentangle": bias_disentangle.evaluate_saved_adapter,
}


def get_evaluator(name: str) -> Callable[..., dict[str, float]]:
    lowered = str(name).strip().lower()
    if lowered not in EVALUATORS:
        raise ValueError(f"Unknown evaluator: {name}. Available evaluators: {sorted(EVALUATORS)}")
    return EVALUATORS[lowered]
