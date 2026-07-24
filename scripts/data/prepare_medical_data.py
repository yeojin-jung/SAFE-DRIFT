#!/usr/bin/env python3
"""
Build a MedQA-target / MedMCQA-candidate real-data experiment for SAFE selection.

Output layout:
  - target_train.jsonl: small gold MedQA train subset used for target gradients
  - target_dev.jsonl: full MedQA dev split for model selection / development
  - target_test.jsonl: full MedQA test split for final reporting
  - candidate_pool.jsonl: candidate pool from MedMCQA train or MedQA-train remainder
  - reference_prompts.jsonl: labeled off-target records suitable for reference Fisher
  - reference_eval/*.jsonl: benchmark-specific labeled or prompt-only eval files
  - reference_eval/mcq_ood_test.jsonl: held-out multiple-choice off-target test bundle
  - manifest.json: experiment metadata and counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


MEDQA_SOURCE = "davidheineman/medqa-en"
MEDMCQA_SOURCE = "openlifescienceai/medmcqa"
GSM8K_SOURCE = ("openai/gsm8k", "main")
MMLU_SOURCE = ("cais/mmlu", "all")
IFEVAL_SOURCE = "google/IFEval"
TRUTHFULQA_CANDIDATES = [
    ("truthful_qa", ("generation",), "validation"),
    ("domenicrosati/TruthfulQA", tuple(), "validation"),
]
BBQ_SOURCE = "heegyu/bbq"
BBQ_CONFIGS = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation",
]
DEFAULT_REFERENCE_DATASETS = [
    "gsm8k",
    "mmlu_non_medical",
    "ifeval",
    "truthfulqa",
    "bbq",
]
MMLU_MEDICAL_SUBJECTS = {
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "nutrition",
    "professional_medicine",
    "human_aging",
    "virology",
}


@dataclass
class BuildConfig:
    out_dir: str
    seed: int
    max_target_train: int
    max_candidates: int
    max_candidate_validation: int
    max_candidate_test: int
    max_reference_per_dataset: int
    max_reference_validation_per_dataset: int
    max_reference_test_per_dataset: int
    reference_fisher_max_total: int | None
    reference_mixture_total_budget: int | None
    target_sample_seed: int | None
    legacy_reuse_reference_for_eval: bool
    reference_datasets: list[str]
    eval_only_datasets: list[str]
    reference_mixtures: list[dict[str, Any]]
    candidate_source: str
    overwrite: bool


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def safe_slug(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text).strip("_") or "value"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "_comma_": ",",
        "_period_": ".",
        "_question_": "?",
        "_exclamation_": "!",
        "_apostrophe_": "'",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def load_hf_dataset(repo: str, *args: Any, split: str, **kwargs: Any):
    if load_dataset is None:
        raise RuntimeError(
            "Could not import `datasets`. Install the environment first.\n"
            f"Original import error: {DATASETS_IMPORT_ERROR}"
        )
    kwargs.setdefault("trust_remote_code", True)
    try:
        return load_dataset(repo, *args, split=split, **kwargs)
    except Exception as exc:
        if "trust_remote_code=True" in str(exc):
            kwargs["trust_remote_code"] = True
            return load_dataset(repo, *args, split=split, **kwargs)
        raise RuntimeError(f"Failed to load {repo!r} split={split!r} args={args}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_records(records: Sequence[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    if max_count <= 0 or len(records) <= max_count:
        return list(records)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in indices[:max_count]]


def shuffle_records(records: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return ordered


def split_primary_validation_test(
    records: Sequence[dict[str, Any]],
    *,
    primary_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = [record for record in records if record is not None]
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    ordered = [records[index] for index in indices]
    validation_count = max(0, int(validation_count))
    test_count = max(0, int(test_count))
    if primary_count <= 0:
        primary_count = max(0, len(ordered) - validation_count - test_count)
    primary_count = max(0, int(primary_count))
    primary = ordered[:primary_count]
    validation = ordered[primary_count : primary_count + validation_count]
    test = ordered[primary_count + validation_count : primary_count + validation_count + test_count]
    return primary, validation, test


def parse_reference_mixture_specs(raw: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in [part.strip() for part in str(raw or "").split(",") if part.strip()]:
        if "=" not in item:
            raise ValueError(
                "Reference mixture specs must use NAME=dataset+dataset syntax; "
                f"got {item!r}."
            )
        name, dataset_expr = item.split("=", 1)
        datasets = [part.strip() for part in dataset_expr.split("+") if part.strip()]
        if not name.strip() or len(datasets) < 1:
            raise ValueError(f"Invalid reference mixture spec: {item!r}")
        specs.append({"name": name.strip(), "datasets": datasets})
    return specs


def balanced_domain_counts(total_budget: int, domains: Sequence[str]) -> dict[str, int]:
    if total_budget <= 0:
        raise ValueError("reference_mixture_total_budget must be positive.")
    if not domains:
        raise ValueError("Reference mixtures require at least one domain.")
    base = int(total_budget) // len(domains)
    remainder = int(total_budget) % len(domains)
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }


def with_reference_mixture_metadata(
    record: dict[str, Any],
    *,
    mixture_name: str,
    domains: Sequence[str],
    domain: str,
    domain_count: int,
) -> dict[str, Any]:
    if domain_count <= 0:
        raise ValueError("domain_count must be positive.")
    out = dict(record)
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "reference_mixture": mixture_name,
            "reference_mixture_domains": list(domains),
            "reference_mixture_domain": domain,
            "reference_mixture_domain_count": int(domain_count),
            "reference_mixture_domain_weight": 1.0 / float(len(domains)),
        }
    )
    out["metadata"] = metadata
    out["reference_mixture"] = mixture_name
    out["reference_mixture_domains"] = list(domains)
    out["reference_fisher_domain"] = domain
    out["reference_fisher_weight"] = 1.0 / (float(len(domains)) * float(domain_count))
    return out


def build_reference_mixture_records(
    *,
    records_by_dataset: dict[str, list[dict[str, Any]]],
    mixture_name: str,
    domains: Sequence[str],
    total_budget: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    per_domain_counts = balanced_domain_counts(total_budget, domains)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for domain in domains:
        available = records_by_dataset.get(domain, [])
        desired = per_domain_counts[domain]
        if len(available) < desired:
            raise ValueError(
                f"Reference mixture {mixture_name!r} requested {desired} examples "
                f"from {domain!r}, but only {len(available)} supervised fit examples are available."
            )
        selected = sample_records(
            available,
            desired,
            seed + stable_int(f"reference_mixture:{mixture_name}:{domain}"),
        )
        counts[domain] = len(selected)
        records.extend(
            with_reference_mixture_metadata(
                record,
                mixture_name=mixture_name,
                domains=domains,
                domain=domain,
                domain_count=len(selected),
            )
            for record in selected
        )
    return shuffle_records(records, seed + stable_int(f"reference_mixture:{mixture_name}")), counts


def choice_labels(num_choices: int) -> list[str]:
    if num_choices <= 0 or num_choices > 26:
        raise ValueError(f"Unsupported number of choices: {num_choices}")
    return [chr(ord("A") + idx) for idx in range(num_choices)]


def choice_prompt(question: str, choices: Sequence[str]) -> str:
    labels = choice_labels(len(choices))
    lines = [
        "Question:",
        clean_text(question),
        "",
        "Choices:",
    ]
    for label, choice in zip(labels, choices, strict=True):
        lines.append(f"{label}. {clean_text(choice)}")
    lines.extend(["", "Answer with the correct option only."])
    return "\n".join(lines)


def record_id(prefix: str, source: str, split: str, text: str) -> str:
    return f"{prefix}-{source}-{split}-{stable_hash(text)}"


def make_record(
    *,
    source: str,
    split: str,
    task: str,
    prompt: str,
    assistant: str | None,
    role: str,
    metadata: dict[str, Any] | None = None,
    choices: list[str] | None = None,
    answer: Any = None,
    answer_index: int | None = None,
) -> dict[str, Any] | None:
    prompt = str(prompt).strip()
    if len(prompt) < 5:
        return None
    messages = [{"role": "user", "content": prompt}]
    assistant_text = clean_text(assistant) if assistant is not None else ""
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
    key_text = prompt + "\n" + assistant_text
    return {
        "id": record_id(task, source, split, key_text),
        "source": source,
        "split": split,
        "task": task,
        "role": role,
        "prompt": prompt,
        "messages": messages,
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "metadata": metadata or {},
        "text_hash": stable_hash(key_text, length=24),
    }


def has_supervised_target(record: dict[str, Any]) -> bool:
    return len(record.get("messages", [])) >= 2 and bool(clean_text(record["messages"][-1].get("content", "")))


def gsm8k_final_answer(answer: str) -> str:
    text = clean_text(answer)
    if "####" in text:
        text = text.split("####", 1)[1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text


def format_medqa_split(split: str, role: str) -> list[dict[str, Any]]:
    ds = load_hf_dataset(MEDQA_SOURCE, split=split)
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(ds):
        question = clean_text(row.get("question", ""))
        choices = [clean_text(choice) for choice in row.get("choices", []) if clean_text(choice)]
        answer_idx = row.get("answer_idx")
        if not question or not choices or answer_idx is None:
            continue
        answer_idx = int(answer_idx)
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
        prompt = choice_prompt(question, choices)
        answer_letter = choice_labels(len(choices))[answer_idx]
        record = make_record(
            source="medqa_en",
            split=split,
            task="medical_qa",
            prompt=prompt,
            assistant=answer_letter,
            role=role,
            metadata={
                "row_idx": row_idx,
                "answer_text": clean_text(row.get("answer", "")),
                "answer_idx": answer_idx,
                "meta_info": clean_text(row.get("meta_info", "")),
            },
            choices=choices,
            answer=clean_text(row.get("answer", "")),
            answer_index=answer_idx,
        )
        if record:
            records.append(record)
    return records


def format_medmcqa_train() -> list[dict[str, Any]]:
    ds = load_hf_dataset(MEDMCQA_SOURCE, split="train")
    records: list[dict[str, Any]] = []
    choice_keys = ["opa", "opb", "opc", "opd"]
    for row in ds:
        question = clean_text(row.get("question", ""))
        choices = [clean_text(row.get(key, "")) for key in choice_keys]
        choices = [choice for choice in choices if choice]
        answer_idx = row.get("cop")
        if not question or len(choices) < 2 or answer_idx is None:
            continue
        answer_idx = int(answer_idx)
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
        prompt = choice_prompt(question, choices)
        answer_letter = choice_labels(len(choices))[answer_idx]
        record = make_record(
            source="medmcqa",
            split="train",
            task="medical_qa",
            prompt=prompt,
            assistant=answer_letter,
            role="candidate",
            metadata={
                "id": row.get("id"),
                "subject_name": clean_text(row.get("subject_name", "")),
                "topic_name": clean_text(row.get("topic_name", "")),
                "choice_type": clean_text(row.get("choice_type", "")),
                "explanation": clean_text(row.get("exp", "")),
            },
            choices=choices,
            answer=choices[answer_idx],
            answer_index=answer_idx,
        )
        if record:
            records.append(record)
    return records


def format_gsm8k(max_count: int, seed: int) -> list[dict[str, Any]]:
    ds = load_hf_dataset(GSM8K_SOURCE[0], GSM8K_SOURCE[1], split="test")
    rows = sample_records(list(ds), max_count, seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        question = clean_text(row.get("question", ""))
        answer = gsm8k_final_answer(str(row.get("answer", "")))
        record = make_record(
            source="gsm8k",
            split="test",
            task="math_reasoning",
            prompt="Solve the following grade-school math problem. Give the final numeric answer only.\n\n"
            + question,
            assistant=answer,
            role="reference",
            metadata={"question": question},
            answer=answer,
        )
        if record:
            records.append(record)
    return records


def format_mmlu_non_medical(max_count: int, seed: int) -> list[dict[str, Any]]:
    ds = load_hf_dataset(MMLU_SOURCE[0], MMLU_SOURCE[1], split="validation")
    filtered_rows = [
        row
        for row in ds
        if clean_text(row.get("subject", "")).lower() not in MMLU_MEDICAL_SUBJECTS
    ]
    rows = sample_records(filtered_rows, max_count, seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        question = clean_text(row.get("question", ""))
        choices = [clean_text(choice) for choice in row.get("choices", []) if clean_text(choice)]
        answer_idx = row.get("answer")
        if not question or not choices or answer_idx is None:
            continue
        answer_idx = int(answer_idx)
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
        prompt = choice_prompt(question, choices)
        answer_letter = choice_labels(len(choices))[answer_idx]
        record = make_record(
            source="mmlu_non_medical",
            split="validation",
            task="mmlu_reasoning",
            prompt=prompt,
            assistant=answer_letter,
            role="reference",
            metadata={"subject": clean_text(row.get("subject", ""))},
            choices=choices,
            answer=choices[answer_idx],
            answer_index=answer_idx,
        )
        if record:
            records.append(record)
    return records


def format_ifeval(max_count: int, seed: int) -> list[dict[str, Any]]:
    ds = load_hf_dataset(IFEVAL_SOURCE, split="train")
    rows = sample_records(list(ds), max_count, seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        record = make_record(
            source="ifeval",
            split="train",
            task="instruction_following",
            prompt=clean_text(row.get("prompt", "")),
            assistant=None,
            role="reference_eval",
            metadata={
                "key": row.get("key"),
                "instruction_id_list": row.get("instruction_id_list"),
                "kwargs": row.get("kwargs"),
            },
        )
        if record:
            records.append(record)
    return records


def format_truthfulqa(max_count: int, seed: int) -> list[dict[str, Any]]:
    ds = None
    used_repo = None
    used_split = None
    errors: list[str] = []
    for repo, args, split in TRUTHFULQA_CANDIDATES:
        try:
            ds = load_hf_dataset(repo, *args, split=split)
            used_repo = repo
            used_split = split
            break
        except Exception as exc:
            errors.append(f"{repo}: {exc}")
    if ds is None:
        raise RuntimeError("Failed to load TruthfulQA from known locations:\n" + "\n".join(errors))

    rows = sample_records(list(ds), max_count, seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        question = clean_text(row.get("question") or row.get("Question") or "")
        answer = row.get("best_answer") or row.get("Best Answer") or row.get("correct_answers") or row.get("Correct Answers")
        assistant = ""
        if isinstance(answer, str):
            assistant = clean_text(answer)
        elif isinstance(answer, (list, tuple)):
            cleaned = [clean_text(item) for item in answer if clean_text(item)]
            if cleaned:
                assistant = cleaned[0]
        record = make_record(
            source="truthfulqa",
            split=used_split or "validation",
            task="truthfulness",
            prompt="Answer the following question truthfully and concisely.\n\n" + question,
            assistant=assistant or None,
            role="reference",
            metadata={
                "repo": used_repo,
                "category": clean_text(row.get("category") or row.get("Category") or ""),
            },
            answer=assistant,
        )
        if record:
            records.append(record)
    return records


def format_bbq(max_count: int, seed: int) -> list[dict[str, Any]]:
    per_config = max(1, int(max_count / max(1, len(BBQ_CONFIGS))))
    records: list[dict[str, Any]] = []
    for config_name in BBQ_CONFIGS:
        ds = None
        for split in ("test", "train"):
            try:
                ds = load_hf_dataset(BBQ_SOURCE, config_name, split=split)
                break
            except Exception:
                ds = None
        if ds is None:
            continue
        rows = sample_records(list(ds), per_config, seed + stable_int(config_name))
        for row in rows:
            context = clean_text(row.get("context", ""))
            question = clean_text(row.get("question", ""))
            choices = [clean_text(row.get(f"ans{idx}", "")) for idx in range(3)]
            choices = [choice for choice in choices if choice]
            label = row.get("label")
            if not choices or label is None:
                continue
            answer_idx = int(label)
            if answer_idx < 0 or answer_idx >= len(choices):
                continue
            prompt = choice_prompt((context + "\n\n" + question).strip(), choices)
            answer_letter = choice_labels(len(choices))[answer_idx]
            record = make_record(
                source="bbq",
                split="test",
                task="bias_qa",
                prompt=prompt,
                assistant=answer_letter,
                role="reference",
                metadata={
                    "category": config_name,
                    "context_condition": row.get("context_condition"),
                    "question_polarity": row.get("question_polarity"),
                },
                choices=choices,
                answer=choices[answer_idx],
                answer_index=answer_idx,
            )
            if record:
                records.append(record)
    return sample_records(records, max_count, seed + 777)


def build_reference_artifacts(out_dir: Path, config: BuildConfig) -> dict[str, int]:
    builders = {
        "gsm8k": format_gsm8k,
        "mmlu_non_medical": format_mmlu_non_medical,
        "ifeval": format_ifeval,
        "truthfulqa": format_truthfulqa,
        "bbq": format_bbq,
    }
    eval_dir = out_dir / "reference_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    fisher_records: list[dict[str, Any]] = []
    reference_validation_records: list[dict[str, Any]] = []
    reference_test_records: list[dict[str, Any]] = []
    mcq_ood_validation_records: list[dict[str, Any]] = []
    mcq_ood_test_records: list[dict[str, Any]] = []
    fit_records_by_dataset: dict[str, list[dict[str, Any]]] = {}

    all_datasets = list(dict.fromkeys(config.reference_datasets + config.eval_only_datasets))
    for name in all_datasets:
        if name not in builders:
            raise ValueError(f"Unknown reference dataset {name!r}. Choose from {sorted(builders)}.")
        print(f"[reference] formatting {name} ...", file=sys.stderr)
        eval_only = name in config.eval_only_datasets
        fit_count = 0 if eval_only else int(config.max_reference_per_dataset)
        total_count = fit_count
        if not config.legacy_reuse_reference_for_eval:
            total_count += (
                int(config.max_reference_validation_per_dataset)
                + int(config.max_reference_test_per_dataset)
            )
        records = builders[name](total_count, config.seed + stable_int(name))
        if eval_only:
            ordered = shuffle_records(
                records,
                config.seed + stable_int(f"{name}:eval_only_split"),
            )
            validation_count = max(0, int(config.max_reference_validation_per_dataset))
            test_count = max(0, int(config.max_reference_test_per_dataset))
            reference_fit = []
            reference_validation = ordered[:validation_count]
            reference_test = ordered[validation_count : validation_count + test_count]
        elif config.legacy_reuse_reference_for_eval:
            reference_fit = list(records)
            reference_validation = list(records)
            reference_test = list(records)
        else:
            reference_fit, reference_validation, reference_test = split_primary_validation_test(
                records,
                primary_count=fit_count,
                validation_count=config.max_reference_validation_per_dataset,
                test_count=config.max_reference_test_per_dataset,
                seed=config.seed + stable_int(f"{name}:reference_split"),
            )
        counts[f"reference_source/{name}/fit"] = len(reference_fit)
        counts[f"reference_source/{name}/validation"] = write_jsonl(
            eval_dir / f"{name}_validation.jsonl",
            reference_validation,
        )
        counts[f"reference_source/{name}/test"] = write_jsonl(eval_dir / f"{name}_test.jsonl", reference_test)
        counts[f"reference_eval/{name}"] = write_jsonl(eval_dir / f"{name}.jsonl", reference_test)
        if name in {"mmlu_non_medical", "bbq"}:
            mcq_ood_validation_records.extend(reference_validation)
            mcq_ood_test_records.extend(reference_test)
        if eval_only:
            continue
        supervised_fit = [record for record in reference_fit if has_supervised_target(record)]
        fit_records_by_dataset[name] = supervised_fit
        if name == "ifeval":
            reference_validation_records.extend(reference_validation)
            reference_test_records.extend(reference_test)
            continue
        fisher_records.extend(supervised_fit)
        reference_validation_records.extend([record for record in reference_validation if has_supervised_target(record)])
        reference_test_records.extend([record for record in reference_test if has_supervised_target(record)])

    supervised_fisher_pool = list(fisher_records)
    if config.reference_fisher_max_total is not None:
        fisher_records = sample_records(
            supervised_fisher_pool,
            config.reference_fisher_max_total,
            config.seed + stable_int("reference_prompts"),
        )

    counts["reference_prompt_supervised_pool"] = len(supervised_fisher_pool)
    counts["reference_prompts"] = write_jsonl(out_dir / "reference_prompts.jsonl", fisher_records)
    reference_validation_records = shuffle_records(
        reference_validation_records,
        config.seed + stable_int("reference_validation_combined"),
    )
    reference_test_records = shuffle_records(
        reference_test_records,
        config.seed + stable_int("reference_test_combined"),
    )
    mcq_ood_validation_records = shuffle_records(
        mcq_ood_validation_records,
        config.seed + stable_int("mcq_ood_validation_combined"),
    )
    mcq_ood_test_records = shuffle_records(
        mcq_ood_test_records,
        config.seed + stable_int("mcq_ood_test_combined"),
    )
    counts["reference_validation"] = write_jsonl(out_dir / "reference_validation.jsonl", reference_validation_records)
    counts["reference_test"] = write_jsonl(out_dir / "reference_test.jsonl", reference_test_records)
    counts["reference_eval/mcq_ood_validation"] = write_jsonl(
        eval_dir / "mcq_ood_validation.jsonl",
        mcq_ood_validation_records,
    )
    counts["reference_eval/mcq_ood_test"] = write_jsonl(eval_dir / "mcq_ood_test.jsonl", mcq_ood_test_records)
    counts["reference_eval/mcq_ood_eval"] = write_jsonl(eval_dir / "mcq_ood_eval.jsonl", mcq_ood_test_records)
    if config.reference_mixtures:
        if config.reference_mixture_total_budget is None:
            raise ValueError("--reference-mixtures requires --reference-mixture-total-budget.")
        mixture_dir = out_dir / "reference_mixtures"
        mixture_dir.mkdir(parents=True, exist_ok=True)
        for spec in config.reference_mixtures:
            mixture_name = str(spec["name"])
            domains = [str(domain) for domain in spec["datasets"]]
            unknown = [domain for domain in domains if domain not in fit_records_by_dataset]
            if unknown:
                raise ValueError(
                    f"Reference mixture {mixture_name!r} uses datasets that were not "
                    f"prepared for Fisher fitting: {unknown}."
                )
            mixture_records, per_domain_counts = build_reference_mixture_records(
                records_by_dataset=fit_records_by_dataset,
                mixture_name=mixture_name,
                domains=domains,
                total_budget=int(config.reference_mixture_total_budget),
                seed=config.seed,
            )
            mixture_slug = safe_slug(mixture_name)
            counts[f"reference_mixture/{mixture_slug}"] = write_jsonl(
                mixture_dir / f"{mixture_slug}.jsonl",
                mixture_records,
            )
            for domain, count in per_domain_counts.items():
                counts[f"reference_mixture/{mixture_slug}/{domain}"] = count
    return counts


def build_medical_dataset(out_dir: Path, config: BuildConfig) -> dict[str, int]:
    print("[medical] loading MedQA splits ...", file=sys.stderr)
    medqa_train_all = format_medqa_split("train", role="target_gradient")
    target_seed = config.seed + 1 if config.target_sample_seed is None else config.target_sample_seed
    target_train = sample_records(medqa_train_all, config.max_target_train, target_seed)
    target_train_ids = {record["id"] for record in target_train}

    print("[medical] loading MedQA dev/test ...", file=sys.stderr)
    target_dev = shuffle_records(
        format_medqa_split("dev", role="target_eval_dev"),
        config.seed + stable_int("target_dev"),
    )
    target_test = shuffle_records(
        format_medqa_split("test", role="target_eval_test"),
        config.seed + stable_int("target_test"),
    )

    if config.candidate_source == "medmcqa":
        print("[medical] loading MedMCQA train candidates ...", file=sys.stderr)
        candidate_records = format_medmcqa_train()
    elif config.candidate_source == "medqa_train_remainder":
        print("[medical] sampling MedQA train remainder as candidates ...", file=sys.stderr)
        candidate_records = [record for record in medqa_train_all if record["id"] not in target_train_ids]
    else:
        raise ValueError(f"Unsupported candidate_source: {config.candidate_source}")
    candidates, candidate_validation, candidate_test = split_primary_validation_test(
        candidate_records,
        primary_count=config.max_candidates,
        validation_count=config.max_candidate_validation,
        test_count=config.max_candidate_test,
        seed=config.seed + 2,
    )

    counts = {
        "target_train": write_jsonl(out_dir / "target_train.jsonl", target_train),
        "target_dev": write_jsonl(out_dir / "target_dev.jsonl", target_dev),
        "target_test": write_jsonl(out_dir / "target_test.jsonl", target_test),
        "candidate_pool": write_jsonl(out_dir / "candidate_pool.jsonl", candidates),
        "candidate_validation": write_jsonl(out_dir / "candidate_validation.jsonl", candidate_validation),
        "candidate_test": write_jsonl(out_dir / "candidate_test.jsonl", candidate_test),
    }
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MedQA-target / MedMCQA-candidate real-data SAFE experiment."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-target-train", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--max-candidate-validation", type=int, default=0)
    parser.add_argument("--max-candidate-test", type=int, default=0)
    parser.add_argument("--max-reference-per-dataset", type=int, default=256)
    parser.add_argument("--max-reference-validation-per-dataset", type=int, default=0)
    parser.add_argument("--max-reference-test-per-dataset", type=int, default=0)
    parser.add_argument("--reference-fisher-max-total", type=int, default=None)
    parser.add_argument(
        "--reference-mixture-total-budget",
        type=int,
        default=None,
        help="Fixed total Fisher examples per named reference mixture.",
    )
    parser.add_argument("--target-sample-seed", type=int, default=None)
    parser.add_argument("--legacy-reuse-reference-for-eval", action="store_true")
    parser.add_argument(
        "--reference-datasets",
        default=",".join(DEFAULT_REFERENCE_DATASETS),
        help="Comma-separated list from: gsm8k,mmlu_non_medical,ifeval,truthfulqa,bbq",
    )
    parser.add_argument(
        "--eval-only-datasets",
        default="",
        help=(
            "Comma-separated non-medical datasets prepared only for held-out evaluation, "
            "never for the reference Fisher."
        ),
    )
    parser.add_argument(
        "--reference-mixtures",
        default="",
        help=(
            "Comma-separated NAME=dataset+dataset specs. Each mixture is written under "
            "reference_mixtures/ with exact per-domain Fisher weights."
        ),
    )
    parser.add_argument(
        "--candidate-source",
        choices=["medmcqa", "medqa_train_remainder"],
        default="medmcqa",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory {out_dir} already exists and is non-empty. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BuildConfig(
        out_dir=str(out_dir),
        seed=args.seed,
        max_target_train=args.max_target_train,
        max_candidates=args.max_candidates,
        max_candidate_validation=args.max_candidate_validation,
        max_candidate_test=args.max_candidate_test,
        max_reference_per_dataset=args.max_reference_per_dataset,
        max_reference_validation_per_dataset=args.max_reference_validation_per_dataset,
        max_reference_test_per_dataset=args.max_reference_test_per_dataset,
        reference_fisher_max_total=args.reference_fisher_max_total,
        reference_mixture_total_budget=args.reference_mixture_total_budget,
        target_sample_seed=args.target_sample_seed,
        legacy_reuse_reference_for_eval=bool(args.legacy_reuse_reference_for_eval),
        reference_datasets=[item.strip() for item in args.reference_datasets.split(",") if item.strip()],
        eval_only_datasets=[item.strip() for item in args.eval_only_datasets.split(",") if item.strip()],
        reference_mixtures=parse_reference_mixture_specs(args.reference_mixtures),
        candidate_source=args.candidate_source,
        overwrite=bool(args.overwrite),
    )

    print(f"[build] medical experiment out={out_dir}", file=sys.stderr)
    counts = build_medical_dataset(out_dir, config)
    counts.update(build_reference_artifacts(out_dir, config))

    manifest = {
        "config": asdict(config),
        "sources": {
            "target": MEDQA_SOURCE,
            "candidate": MEDMCQA_SOURCE if config.candidate_source == "medmcqa" else MEDQA_SOURCE,
            "references": {
                "gsm8k": GSM8K_SOURCE[0],
                "mmlu_non_medical": MMLU_SOURCE[0],
                "ifeval": IFEVAL_SOURCE,
                "truthfulqa": [candidate[0] for candidate in TRUTHFULQA_CANDIDATES],
                "bbq": BBQ_SOURCE,
            },
        },
        "counts": counts,
        "schema": {
            "target_gradient_file": "target_train.jsonl",
            "target_dev_file": "target_dev.jsonl",
            "target_test_file": "target_test.jsonl",
            "candidate_pool_file": "candidate_pool.jsonl",
            "candidate_validation_file": "candidate_validation.jsonl",
            "candidate_test_file": "candidate_test.jsonl",
            "reference_prompt_file": "reference_prompts.jsonl",
            "reference_mixture_dir": "reference_mixtures/",
            "reference_validation_file": "reference_validation.jsonl",
            "reference_test_file": "reference_test.jsonl",
            "reference_eval_dir": "reference_eval/",
            "reference_ood_eval_file": "reference_eval/mcq_ood_test.jsonl",
            "fisher_reference_excludes": ["ifeval"],
            "supervised_answer_format": "assistant contains only the correct option letter for MCQ tasks",
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"out_dir": str(out_dir), "counts": counts}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
