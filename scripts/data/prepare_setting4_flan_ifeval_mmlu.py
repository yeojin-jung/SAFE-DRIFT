#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
DEFAULT_IFEVAL_PROXY_OUTPUT = "I will follow all instructions in the prompt."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_no}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def load_hf_dataset(dataset: str, *, name: str | None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install datasets or pass a local JSONL source file.") from exc
    if name:
        loaded = load_dataset(dataset, name, split=split)
    else:
        loaded = load_dataset(dataset, split=split)
    return [dict(row) for row in loaded]


def sample_records(records: list[dict[str, Any]], max_count: int | None, seed: int) -> list[dict[str, Any]]:
    if max_count is None or len(records) <= max_count:
        return list(records)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max_count))
    return [records[index] for index in indices]


def take_records(records: list[dict[str, Any]], max_count: int | None) -> list[dict[str, Any]]:
    if max_count is None:
        return list(records)
    return list(records[: max(0, int(max_count))])


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            if isinstance(message, dict) and message.get("role"):
                normalized.append({"role": str(message["role"]), "content": str(message.get("content", ""))})
        if normalized:
            return normalized
    prompt = str(record.get("instruction", record.get("prompt", record.get("input", "")))).strip()
    output = str(record.get("output", record.get("answer", ""))).strip()
    messages = []
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if output:
        messages.append({"role": "assistant", "content": output})
    if not messages:
        raise ValueError(f"Cannot normalize record without prompt/messages: {record}")
    return messages


def normalize_candidate_records(records: list[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    selected = sample_records(records, max_count, seed)
    normalized = []
    for idx, record in enumerate(selected):
        normalized.append(
            {
                **record,
                "id": record.get("id", f"flan_v2_{idx}"),
                "source": record.get("source", record.get("dataset", "flan_v2")),
                "messages": normalize_messages(record),
            }
        )
    return normalized


def ifeval_prompt(record: dict[str, Any]) -> str:
    prompt = str(record.get("prompt", "")).strip()
    if prompt:
        return prompt
    for message in normalize_messages(record):
        if message["role"] == "user" and message["content"].strip():
            return message["content"].strip()
    raise ValueError(f"IFEval record has no prompt: {record}")


def normalize_ifeval_prompt_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    prompt = ifeval_prompt(record)
    return {
        "id": record.get("id", record.get("key", idx)),
        "key": record.get("key", record.get("id", idx)),
        "dataset": "ifeval",
        "instruction_id_list": list(record.get("instruction_id_list", record.get("instruction_ids", []))),
        "kwargs": list(record.get("kwargs", record.get("instruction_kwargs", []))),
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
    }


def normalize_ifeval_target_record(record: dict[str, Any], idx: int, proxy_output: str) -> dict[str, Any]:
    prompt_only = normalize_ifeval_prompt_record(record, idx)
    assistant_outputs = [
        message["content"]
        for message in normalize_messages(record)
        if message["role"] == "assistant" and str(message["content"]).strip()
    ]
    output = assistant_outputs[-1] if assistant_outputs else str(record.get("output", "")).strip()
    if not output:
        output = proxy_output
    return {
        **prompt_only,
        "messages": [
            {"role": "user", "content": prompt_only["prompt"]},
            {"role": "assistant", "content": output},
        ],
    }


def format_mmlu_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    if isinstance(record.get("messages"), list):
        return {
            **record,
            "id": record.get("id", f"mmlu_{idx}"),
            "source": record.get("source", "mmlu"),
        }
    question = str(record.get("question", record.get("prompt", ""))).strip()
    choices = list(record.get("choices", []))
    answer = record.get("answer")
    if isinstance(answer, str) and answer.strip().upper() in CHOICE_LABELS:
        answer_label = answer.strip().upper()
    else:
        answer_label = CHOICE_LABELS[int(answer)]
    choice_lines = [f"{CHOICE_LABELS[i]}. {choice}" for i, choice in enumerate(choices)]
    subject = str(record.get("subject", "mmlu"))
    prompt = "\n".join(
        [
            f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.",
            "",
            question,
            *choice_lines,
            "Answer:",
        ]
    )
    return {
        "id": record.get("id", f"{subject}_{idx}"),
        "source": "mmlu",
        "subject": subject,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"The answer is: {answer_label}"},
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Setting 4 FLAN-v2 / IFEval / MMLU JSONL files.")
    parser.add_argument("--out", default="data/setting_4_flan_ifeval_mmlu/prepared")
    parser.add_argument("--flan-source", default="data/raw/flan_v2_data.jsonl")
    parser.add_argument("--ifeval-source", default=None)
    parser.add_argument("--ifeval-hf-dataset", default="google/IFEval")
    parser.add_argument("--ifeval-hf-split", default="train")
    parser.add_argument("--mmlu-source", default=None, help="Optional source used for both MMLU reference and eval files.")
    parser.add_argument("--mmlu-reference-source", default=None)
    parser.add_argument("--mmlu-eval-source", default=None)
    parser.add_argument("--mmlu-hf-dataset", default="cais/mmlu")
    parser.add_argument("--mmlu-hf-name", default="all")
    parser.add_argument("--mmlu-reference-hf-split", default="validation")
    parser.add_argument("--mmlu-eval-hf-split", default="test")
    parser.add_argument("--max-candidates", type=int, default=100000)
    parser.add_argument("--max-ifeval-target", type=int, default=42)
    parser.add_argument("--max-ifeval-eval", type=int, default=491)
    parser.add_argument("--max-mmlu-reference", type=int, default=285)
    parser.add_argument("--max-mmlu-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ifeval-proxy-output", default=DEFAULT_IFEVAL_PROXY_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    outputs = {
        "candidate_file": out / "flan_v2_100k.jsonl",
        "target_file": out / "ifeval_validation_teacher_42.jsonl",
        "validation_file": out / "ifeval_validation_teacher_metrics_eval.jsonl",
        "eval_file": out / "ifeval_test_prompt_only.jsonl",
        "ood_eval_file": out / "mmlu_test_200.jsonl",
        "reference_file": out / "mmlu_validation_285.jsonl",
    }
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing files without --overwrite: {existing}")

    flan_path = Path(args.flan_source)
    if not flan_path.exists():
        raise FileNotFoundError(
            f"Missing FLAN source {flan_path}. Place the FLAN-v2 JSONL there or pass --flan-source."
        )
    flan_records = normalize_candidate_records(read_jsonl(flan_path), args.max_candidates, args.seed)

    if args.ifeval_source:
        ifeval_raw = read_jsonl(Path(args.ifeval_source))
        ifeval_source = str(Path(args.ifeval_source).resolve())
    else:
        ifeval_raw = load_hf_dataset(args.ifeval_hf_dataset, name=None, split=args.ifeval_hf_split)
        ifeval_source = f"hf://{args.ifeval_hf_dataset}/{args.ifeval_hf_split}"
    ifeval_target_raw = sample_records(ifeval_raw, args.max_ifeval_target, args.seed)
    ifeval_eval_raw = take_records(ifeval_raw, args.max_ifeval_eval)
    ifeval_target = [
        normalize_ifeval_target_record(record, idx, args.ifeval_proxy_output)
        for idx, record in enumerate(ifeval_target_raw)
    ]
    ifeval_eval = [normalize_ifeval_prompt_record(record, idx) for idx, record in enumerate(ifeval_eval_raw)]

    mmlu_reference_source_arg = args.mmlu_reference_source or args.mmlu_source
    mmlu_eval_source_arg = args.mmlu_eval_source or args.mmlu_source
    if mmlu_reference_source_arg:
        mmlu_reference_raw = read_jsonl(Path(mmlu_reference_source_arg))
        mmlu_reference_source = str(Path(mmlu_reference_source_arg).resolve())
    else:
        mmlu_reference_raw = load_hf_dataset(args.mmlu_hf_dataset, name=args.mmlu_hf_name, split=args.mmlu_reference_hf_split)
        mmlu_reference_source = f"hf://{args.mmlu_hf_dataset}/{args.mmlu_hf_name}/{args.mmlu_reference_hf_split}"
    if mmlu_eval_source_arg:
        mmlu_eval_raw = read_jsonl(Path(mmlu_eval_source_arg))
        mmlu_eval_source = str(Path(mmlu_eval_source_arg).resolve())
    else:
        mmlu_eval_raw = load_hf_dataset(args.mmlu_hf_dataset, name=args.mmlu_hf_name, split=args.mmlu_eval_hf_split)
        mmlu_eval_source = f"hf://{args.mmlu_hf_dataset}/{args.mmlu_hf_name}/{args.mmlu_eval_hf_split}"
    mmlu_reference = [
        format_mmlu_record(record, idx)
        for idx, record in enumerate(sample_records(mmlu_reference_raw, args.max_mmlu_reference, args.seed))
    ]
    mmlu_eval = [
        format_mmlu_record(record, idx)
        for idx, record in enumerate(sample_records(mmlu_eval_raw, args.max_mmlu_eval, args.seed))
    ]

    write_jsonl(outputs["candidate_file"], flan_records)
    write_jsonl(outputs["target_file"], ifeval_target)
    write_jsonl(outputs["validation_file"], ifeval_target)
    write_jsonl(outputs["eval_file"], ifeval_eval)
    write_jsonl(outputs["ood_eval_file"], mmlu_eval)
    write_jsonl(outputs["reference_file"], mmlu_reference)

    manifest = {
        "seed": args.seed,
        "candidate_description": "FLAN-v2 candidate pool sampled to 100K records by default.",
        "target_description": "IFEval validation prompts converted to supervised selector targets.",
        "eval_description": "Prompt-only IFEval records for final instruction-following evaluation.",
        "ood_eval_description": "Fixed held-out MMLU test slice formatted as multiple-choice chat records.",
        "reference_description": "MMLU validation records used as selector reference Fisher data.",
        "flan_source": str(flan_path.resolve()),
        "ifeval_source": ifeval_source,
        "mmlu_reference_source": mmlu_reference_source,
        "mmlu_eval_source": mmlu_eval_source,
        "candidate_total": len(flan_records),
        "ifeval_target_total": len(ifeval_target),
        "ifeval_eval_total": len(ifeval_eval),
        "mmlu_eval_total": len(mmlu_eval),
        "mmlu_reference_total": len(mmlu_reference),
        **{key: display_path(path) for key, path in outputs.items()},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
