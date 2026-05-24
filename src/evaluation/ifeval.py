from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

from .common import (
    concat_messages_for_generation,
    generate_response_text,
    get_prompt_messages,
)


def _ensure_compat_modules() -> None:
    if "lm_eval.utils" not in sys.modules:
        lm_eval_module = types.ModuleType("lm_eval")
        utils_module = types.ModuleType("lm_eval.utils")
        utils_module.eval_logger = logging.getLogger("safe_drift_ifeval")
        sys.modules.setdefault("lm_eval", lm_eval_module)
        sys.modules.setdefault("lm_eval.utils", utils_module)
    if "pkg_resources" not in sys.modules:
        pkg_resources_module = types.ModuleType("pkg_resources")

        def resource_filename(package_or_requirement: str, resource_name: str) -> str:
            spec = importlib.util.find_spec(package_or_requirement)
            if spec is None or spec.origin is None:
                raise ModuleNotFoundError(package_or_requirement)
            return str(Path(spec.origin).resolve().parent / resource_name)

        pkg_resources_module.resource_filename = resource_filename
        sys.modules["pkg_resources"] = pkg_resources_module


def _candidate_oe_eval_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("SAFE_DRIFT_OE_EVAL_ROOT", "OE_EVAL_ROOT", "OLMES_ROOT"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    repo_root = Path(__file__).resolve().parents[2]
    roots.extend(
        [
            repo_root / "third_party" / "olmes",
            repo_root / "third_party" / "oe-eval",
            repo_root / "third_party" / "oe_eval",
        ]
    )
    return roots


def _load_ifeval_checker():
    _ensure_compat_modules()
    for root in _candidate_oe_eval_roots():
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from oe_eval.dependencies.ifeval.utils import (  # type: ignore
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )
    except ModuleNotFoundError as exc:
        roots = ", ".join(str(path) for path in _candidate_oe_eval_roots())
        raise RuntimeError(
            "IFEval evaluation requires the OLMES/oe-eval IFEval checker. "
            "Install oe-eval in the active environment or set OE_EVAL_ROOT to "
            f"a checkout containing oe_eval/dependencies/ifeval. Checked: {roots}"
        ) from exc
    return InputExample, test_instruction_following_strict, test_instruction_following_loose


def _record_prompt(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt", "")).strip()
    if prompt:
        return prompt
    for message in get_prompt_messages(record):
        if message["role"] == "user" and str(message["content"]).strip():
            return str(message["content"]).strip()
    instruction = str(record.get("instruction", "")).strip()
    if instruction:
        return instruction
    raise ValueError(f"IFEval record has no prompt/user instruction: {record}")


def _input_example(record: dict[str, Any], index: int, InputExample):
    instruction_ids = record.get("instruction_id_list", record.get("instruction_ids"))
    kwargs = record.get("kwargs", record.get("instruction_kwargs"))
    if instruction_ids is None or kwargs is None:
        raise ValueError(
            "IFEval records must include instruction_id_list and kwargs. "
            f"Missing metadata in record {record.get('id', index)!r}."
        )
    if isinstance(kwargs, dict):
        kwargs = [kwargs]
    raw_key = record.get("key", record.get("id", index))
    try:
        key = int(raw_key)
    except (TypeError, ValueError):
        key = index
    return InputExample(
        key=key,
        instruction_id_list=list(instruction_ids),
        prompt=_record_prompt(record),
        kwargs=list(kwargs),
    )


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 512,
    add_bos_token: bool = False,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    if not subset:
        return {
            "ifeval_prompt_level_strict_acc": 0.0,
            "ifeval_prompt_level_loose_acc": 0.0,
            "ifeval_inst_level_strict_acc": 0.0,
            "ifeval_inst_level_loose_acc": 0.0,
            "ifeval_primary_prompt_score": 0.0,
            "ifeval_primary_instruction_score": 0.0,
            "ifeval_count": 0.0,
        }

    InputExample, check_strict, check_loose = _load_ifeval_checker()
    prompt_strict = 0
    prompt_loose = 0
    inst_strict_correct = 0
    inst_loose_correct = 0
    inst_total = 0

    model.eval()
    for index, record in enumerate(subset):
        example = _input_example(record, index, InputExample)
        prompt_text = concat_messages_for_generation(
            get_prompt_messages(record),
            tokenizer=tokenizer,
            add_bos_token=add_bos_token,
        )
        response = generate_response_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        strict = check_strict(example, response)
        loose = check_loose(example, response)
        strict_list = list(strict.follow_instruction_list)
        loose_list = list(loose.follow_instruction_list)
        prompt_strict += int(bool(strict.follow_all_instructions))
        prompt_loose += int(bool(loose.follow_all_instructions))
        inst_strict_correct += sum(bool(value) for value in strict_list)
        inst_loose_correct += sum(bool(value) for value in loose_list)
        inst_total += max(len(strict_list), len(loose_list))

    total = len(subset)
    inst_denominator = max(1, inst_total)
    return {
        "ifeval_prompt_level_strict_acc": float(prompt_strict / total),
        "ifeval_prompt_level_loose_acc": float(prompt_loose / total),
        "ifeval_inst_level_strict_acc": float(inst_strict_correct / inst_denominator),
        "ifeval_inst_level_loose_acc": float(inst_loose_correct / inst_denominator),
        "ifeval_primary_prompt_score": float(prompt_strict / total),
        "ifeval_primary_instruction_score": float(inst_strict_correct / inst_denominator),
        "ifeval_count": float(total),
    }
