#!/usr/bin/env python3
"""Prepare Setting 3: Empathetic Dialogues / ESConv / GSM8K.

Outputs:
  - target_all.jsonl: ESConv target records pooled before the pipeline's 34/33/33 split
  - target_train.jsonl: target-gradient subset retained for inspection/backward compatibility
  - target_eval.jsonl: held-out ESConv subset retained for inspection/backward compatibility
  - candidate_pool.jsonl: Empathetic Dialogues candidate pool
  - reference_prompts.jsonl: GSM8K records for reference Fisher construction
  - reference_validation.jsonl: disjoint GSM8K records for hyperparameter validation
  - reference_test.jsonl: disjoint GSM8K records for held-out drift evaluation
  - reference_eval/gsm8k_test.jsonl: same held-out GSM8K test split for OOD evaluation
  - manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import `datasets`. Install the experiment environment first.\n"
        f"Original import error: {exc}"
    )


ESCONV_SOURCE = "thu-coai/esconv"
EMPATHETIC_DIALOGUES_SOURCE = "facebook/empathetic_dialogues"
GSM8K_SOURCE = "openai/gsm8k"


@dataclass
class BuildConfig:
    out_dir: str
    seed: int
    max_target_train: int
    max_candidates: int
    max_candidate_validation: int
    max_candidate_test: int
    max_target_eval: int
    max_reference_examples: int
    max_reference_validation_examples: int
    max_reference_test_examples: int
    target_splits: list[str]
    overwrite: bool


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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


def record_id(source: str, split: str, text: str, prefix: str) -> str:
    return f"{prefix}-{source}-{split}-{stable_hash(text)}"


def make_sft_record(
    *,
    source: str,
    split: str,
    prompt: str,
    response: str,
    task: str,
    role: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    response = clean_text(response)
    if len(prompt) < 5 or len(response) < 5:
        return None
    text = f"User: {prompt}\nAssistant: {response}"
    return {
        "id": record_id(source, split, prompt + "\n" + response, task),
        "source": source,
        "split": split,
        "task": task,
        "role": role,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "text": text,
        "text_hash": stable_hash(text, length=24),
        "metadata": metadata or {},
    }


def make_prompt_record(
    *,
    source: str,
    split: str,
    prompt: str,
    reference_type: str,
    answer: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    if len(prompt) < 5:
        return None
    messages = [{"role": "user", "content": prompt}]
    answer_text = clean_text(answer)
    if answer_text:
        messages.append({"role": "assistant", "content": answer_text})
    return {
        "id": record_id(source, split, prompt, reference_type),
        "source": source,
        "split": split,
        "task": reference_type,
        "role": "reference",
        "prompt": prompt,
        "messages": messages,
        "answer": answer,
        "text_hash": stable_hash(prompt, length=24),
        "metadata": metadata or {},
    }


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
    records = [record for record in records if record is not None]
    if max_count <= 0 or len(records) <= max_count:
        return list(records)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in indices[:max_count]]


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


def unique_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("id") or record.get("text_hash") or stable_hash(json.dumps(record, sort_keys=True)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("text_hash") or stable_hash(json.dumps(record, sort_keys=True)))


def assert_disjoint_split_sets(named_records: dict[str, Sequence[dict[str, Any]]]) -> dict[str, int]:
    names = list(named_records)
    key_sets = {
        name: {record_key(record) for record in records if record is not None}
        for name, records in named_records.items()
    }
    overlap_counts: dict[str, int] = {}
    examples: list[str] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = key_sets[left_name] & key_sets[right_name]
            check_name = f"{left_name}__{right_name}"
            overlap_counts[check_name] = len(overlap)
            if overlap:
                examples.extend(f"{check_name}:{key}" for key in sorted(overlap)[:5])
    if examples:
        raise ValueError(
            "Prepared data split leakage detected. Overlapping record keys: "
            + ", ".join(examples[:10])
        )
    return overlap_counts


def load_hf_dataset(repo: str, *args: Any, split: str, **kwargs: Any):
    kwargs.setdefault("trust_remote_code", True)
    try:
        return load_dataset(repo, *args, split=split, **kwargs)
    except Exception as exc:
        if "trust_remote_code=True" in str(exc):
            kwargs["trust_remote_code"] = True
            return load_dataset(repo, *args, split=split, **kwargs)
        raise RuntimeError(f"Failed to load {repo!r} split={split!r} args={args}: {exc}") from exc


def parse_esconv_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("text"), str):
        try:
            parsed = json.loads(row["text"])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    if "dialog" in row or "situation" in row:
        return dict(row)
    return None


def speaker_name(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or turn.get("role") or turn.get("from") or turn.get("author") or "").lower()
    if speaker in {"sys", "supporter", "assistant", "gpt", "bot", "therapist", "counselor"}:
        return "assistant"
    if speaker in {"usr", "user", "seeker", "human", "client", "patient"}:
        return "user"
    return speaker or "unknown"


def turn_text(turn: dict[str, Any]) -> str:
    return clean_text(turn.get("text") or turn.get("content") or turn.get("value") or turn.get("utterance") or "")


def format_esconv(split: str) -> list[dict[str, Any]]:
    ds = load_hf_dataset(ESCONV_SOURCE, split=split)
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(ds):
        conv = parse_esconv_row(row)
        if not conv:
            continue
        situation = clean_text(conv.get("situation", ""))
        emotion = clean_text(conv.get("emotion_type", ""))
        problem = clean_text(conv.get("problem_type", ""))
        dialog = conv.get("dialog") or conv.get("conversation") or conv.get("messages") or []
        if not isinstance(dialog, list):
            continue
        history: list[str] = []
        for turn_idx, turn in enumerate(dialog):
            if not isinstance(turn, dict):
                continue
            role = speaker_name(turn)
            text = turn_text(turn)
            if not text:
                continue
            if role == "assistant" and history:
                prompt_parts = []
                if situation:
                    prompt_parts.append(f"Client situation: {situation}")
                if emotion or problem:
                    prompt_parts.append(f"Context: emotion={emotion or 'unknown'}; problem={problem or 'unknown'}.")
                prompt_parts.append("Conversation so far:\n" + "\n".join(history[-8:]))
                prompt_parts.append("Respond as a supportive, reflective counselor. Do not provide a diagnosis.")
                record = make_sft_record(
                    source="esconv",
                    split=split,
                    task="therapy_support",
                    role="target",
                    prompt="\n\n".join(prompt_parts),
                    response=text,
                    metadata={
                        "row_idx": row_idx,
                        "turn_idx": turn_idx,
                        "situation": situation,
                        "emotion_type": emotion,
                        "problem_type": problem,
                        "strategy": turn.get("strategy"),
                    },
                )
                if record:
                    records.append(record)
            display_role = "Client" if role == "user" else "Counselor" if role == "assistant" else "Speaker"
            history.append(f"{display_role}: {text}")
    return records


def format_empathetic_dialogues(split: str) -> list[dict[str, Any]]:
    ds = load_hf_dataset(EMPATHETIC_DIALOGUES_SOURCE, split=split)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        groups[str(row.get("conv_id", "unknown"))].append(dict(row))

    records: list[dict[str, Any]] = []
    for conv_id, rows in groups.items():
        rows = sorted(rows, key=lambda row: int(row.get("utterance_idx") or 0))
        history: list[str] = []
        for row in rows:
            utterance = clean_text(row.get("utterance", ""))
            if not utterance:
                continue
            speaker_idx = row.get("speaker_idx")
            if history:
                context = clean_text(row.get("context", ""))
                original_prompt = clean_text(row.get("prompt", ""))
                prompt_parts = []
                if context:
                    prompt_parts.append(f"Emotional context: {context}.")
                if original_prompt and original_prompt.lower() not in " ".join(history[-3:]).lower():
                    prompt_parts.append(f"Initial situation: {original_prompt}")
                prompt_parts.append("Conversation so far:\n" + "\n".join(history[-6:]))
                prompt_parts.append("Respond empathetically and helpfully.")
                record = make_sft_record(
                    source="empathetic_dialogues",
                    split=split,
                    task="therapy_support",
                    role="candidate",
                    prompt="\n\n".join(prompt_parts),
                    response=utterance,
                    metadata={
                        "conv_id": conv_id,
                        "utterance_idx": row.get("utterance_idx"),
                        "speaker_idx": speaker_idx,
                        "context": context,
                        "selfeval": row.get("selfeval"),
                    },
                )
                if record:
                    records.append(record)
            history.append(f"Speaker {speaker_idx}: {utterance}")
    return records


def split_target_records(
    records: Sequence[dict[str, Any]],
    *,
    train_count: int,
    eval_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    target_train = ordered[: max(0, int(train_count))]
    remaining = ordered[len(target_train) :]
    target_eval = remaining if eval_count <= 0 else remaining[: int(eval_count)]
    return target_train, target_eval, [*target_train, *target_eval]


def format_gsm8k(max_count: int, seed: int) -> list[dict[str, Any]]:
    ds = load_hf_dataset(GSM8K_SOURCE, "main", split="test")
    rows = sample_records(list(ds), max_count, seed)
    records: list[dict[str, Any]] = []
    for row in rows:
        question = clean_text(row.get("question", ""))
        answer = clean_text(row.get("answer", ""))
        prompt = f"Solve the following grade-school math problem. Show your reasoning and give the final answer.\n\n{question}"
        record = make_prompt_record(
            source="gsm8k",
            split="test",
            reference_type="math_reasoning",
            prompt=prompt,
            answer=answer,
            metadata={"question": question},
        )
        if record:
            records.append(record)
    return records


def build_dataset(out_dir: Path, config: BuildConfig) -> tuple[dict[str, int], dict[str, int]]:
    target_records: list[dict[str, Any]] = []
    for split in config.target_splits:
        print(f"[target] formatting ESConv split={split} ...", file=sys.stderr)
        target_records.extend(format_esconv(split))
    target_records = unique_records(target_records)
    target_train, target_eval, target_all = split_target_records(
        target_records,
        train_count=config.max_target_train,
        eval_count=config.max_target_eval,
        seed=config.seed + 1,
    )

    print("[candidate] formatting Empathetic Dialogues split=train ...", file=sys.stderr)
    candidate_records = unique_records(format_empathetic_dialogues("train"))
    candidates, candidate_validation, candidate_test = split_primary_validation_test(
        candidate_records,
        primary_count=config.max_candidates,
        validation_count=config.max_candidate_validation,
        test_count=config.max_candidate_test,
        seed=config.seed + 3,
    )

    print("[reference] formatting GSM8K ...", file=sys.stderr)
    reference_total = (
        int(config.max_reference_examples)
        + int(config.max_reference_validation_examples)
        + int(config.max_reference_test_examples)
    )
    references = format_gsm8k(reference_total, config.seed + stable_int("gsm8k"))
    reference_fit, reference_validation, reference_test = split_primary_validation_test(
        references,
        primary_count=config.max_reference_examples,
        validation_count=config.max_reference_validation_examples,
        test_count=config.max_reference_test_examples,
        seed=config.seed + stable_int("gsm8k:reference_split"),
    )

    overlap_checks: dict[str, int] = {}
    overlap_checks.update(
        assert_disjoint_split_sets(
            {
                "target_train": target_train,
                "target_eval": target_eval,
            }
        )
    )
    overlap_checks.update(
        assert_disjoint_split_sets(
            {
                "candidate_pool": candidates,
                "candidate_validation": candidate_validation,
                "candidate_test": candidate_test,
            }
        )
    )
    overlap_checks.update(
        assert_disjoint_split_sets(
            {
                "reference_fit_gsm8k": reference_fit,
                "reference_validation_gsm8k": reference_validation,
                "reference_test_gsm8k": reference_test,
            }
        )
    )

    counts = {
        "target_all": write_jsonl(out_dir / "target_all.jsonl", target_all),
        "target_train": write_jsonl(out_dir / "target_train.jsonl", target_train),
        "target_eval": write_jsonl(out_dir / "target_eval.jsonl", target_eval),
        "candidate_pool": write_jsonl(out_dir / "candidate_pool.jsonl", candidates),
        "candidate_validation": write_jsonl(out_dir / "candidate_validation.jsonl", candidate_validation),
        "candidate_test": write_jsonl(out_dir / "candidate_test.jsonl", candidate_test),
        "reference_eval/gsm8k_validation": write_jsonl(
            out_dir / "reference_eval" / "gsm8k_validation.jsonl",
            reference_validation,
        ),
        "reference_eval/gsm8k_test": write_jsonl(out_dir / "reference_eval" / "gsm8k_test.jsonl", reference_test),
        "reference_eval/gsm8k": write_jsonl(out_dir / "reference_eval" / "gsm8k.jsonl", reference_test),
        "reference_prompts": write_jsonl(out_dir / "reference_prompts.jsonl", reference_fit),
        "reference_validation": write_jsonl(out_dir / "reference_validation.jsonl", reference_validation),
        "reference_test": write_jsonl(out_dir / "reference_test.jsonl", reference_test),
    }
    return counts, overlap_checks


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-target-train", type=int, default=34)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--max-candidate-validation", type=int, default=0)
    parser.add_argument("--max-candidate-test", type=int, default=0)
    parser.add_argument("--max-target-eval", type=int, default=66)
    parser.add_argument("--max-reference-examples", type=int, default=258)
    parser.add_argument("--max-reference-validation-examples", type=int, default=0)
    parser.add_argument("--max-reference-test-examples", type=int, default=0)
    parser.add_argument("--target-splits", default="validation,test")
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
        max_target_eval=args.max_target_eval,
        max_reference_examples=args.max_reference_examples,
        max_reference_validation_examples=args.max_reference_validation_examples,
        max_reference_test_examples=args.max_reference_test_examples,
        target_splits=parse_csv_list(args.target_splits),
        overwrite=bool(args.overwrite),
    )
    counts, overlap_checks = build_dataset(out_dir, config)
    manifest = {
        "config": asdict(config),
        "sources": {
            "target/esconv": ESCONV_SOURCE,
            "candidate/empathetic_dialogues": EMPATHETIC_DIALOGUES_SOURCE,
            "reference/gsm8k": GSM8K_SOURCE,
        },
        "counts": counts,
        "split_overlap_checks": overlap_checks,
        "schema": {
            "target_file": "target_all.jsonl",
            "candidate_pool_file": "candidate_pool.jsonl",
            "candidate_validation_file": "candidate_validation.jsonl",
            "candidate_test_file": "candidate_test.jsonl",
            "reference_prompt_file": "reference_prompts.jsonl",
            "reference_validation_file": "reference_validation.jsonl",
            "reference_test_file": "reference_test.jsonl",
            "ood_eval_file": "reference_eval/gsm8k_test.jsonl",
            "reference_eval_file": "reference_eval/gsm8k_test.jsonl",
        },
        "reference_split_policy": (
            "GSM8K reference_prompts, reference_validation, and reference_test are sampled "
            "as disjoint splits. The OOD drift evaluator uses reference_eval/gsm8k_test.jsonl, "
            "which is the same held-out split as reference_test.jsonl and is not used for "
            "reference Fisher construction."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": counts}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
