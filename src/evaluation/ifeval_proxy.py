from __future__ import annotations

import json
import re
from typing import Any

import torch

from .common import (
    concat_messages_for_generation,
    generate_response_text,
    get_prompt_messages,
    normalize_text,
)


def normalize_freeform(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"[^0-9a-z\s]+", " ", normalized)
    return " ".join(normalized.split())


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", normalize_freeform(text), flags=re.UNICODE)


def relation_satisfied(value: int, relation: str | None, target: int, *, default_exact: bool) -> bool:
    relation_key = (relation or "").strip().lower()
    if relation_key == "less than":
        return value < target
    if relation_key == "more than":
        return value > target
    if relation_key == "at least":
        return value >= target
    if relation_key == "at most":
        return value <= target
    if relation_key == "exactly":
        return value == target
    return value == target if default_exact else value >= target


def count_sentences(text: str) -> int:
    stripped = str(text).strip()
    if not stripped:
        return 0
    parts = [part.strip() for part in re.split(r"[.!?]+(?:\s+|$)", stripped) if part.strip()]
    return len(parts) if parts else len([line for line in stripped.splitlines() if line.strip()])


def split_paragraphs(text: str) -> list[str]:
    stripped = str(text).strip()
    if not stripped:
        return []
    if "\n\n" in stripped:
        return [chunk.strip() for chunk in re.split(r"\n\s*\n", stripped) if chunk.strip()]
    if re.search(r"\*{3,}", stripped):
        return [chunk.strip() for chunk in re.split(r"\*{3,}", stripped) if chunk.strip()]
    return [line.strip() for line in stripped.splitlines() if line.strip()]


def count_bullets(text: str) -> int:
    return sum(
        bool(re.match(r"^(\*|-|\+)\s+", line.strip()) or re.match(r"^\d+[.)]\s+", line.strip()))
        for line in str(text).splitlines()
    )


def count_keyword_occurrences(text: str, keyword: str) -> int:
    if not keyword:
        return 0
    return len(re.findall(rf"\b{re.escape(keyword.lower())}\b", str(text).lower()))


def check_language_proxy(text: str, language_code: str) -> bool:
    code = str(language_code).strip().lower()
    if not code:
        return True
    script_ranges = {
        "hi": r"[\u0900-\u097F]",
        "mr": r"[\u0900-\u097F]",
        "ne": r"[\u0900-\u097F]",
        "bn": r"[\u0980-\u09FF]",
        "gu": r"[\u0A80-\u0AFF]",
        "kn": r"[\u0C80-\u0CFF]",
        "ta": r"[\u0B80-\u0BFF]",
        "te": r"[\u0C00-\u0C7F]",
        "fa": r"[\u0600-\u06FF]",
        "bg": r"[\u0400-\u04FF]",
    }
    if code in script_ranges:
        return bool(re.search(script_ranges[code], text))
    lowered = normalize_text(text)
    tokens = set(simple_tokens(text))
    if code == "de":
        return bool(tokens & {"und", "der", "die", "das", "ist", "ein", "nicht"} or re.search(r"[äöüß]", lowered))
    if code == "pt":
        return bool(tokens & {"de", "que", "para", "com", "uma", "não", "os", "as"} or re.search(r"[ãõáéíóúç]", lowered))
    if code == "sw":
        return bool(tokens & {"kwa", "na", "ya", "ni", "wa", "katika", "hii"})
    if code == "vi":
        return bool(
            tokens & {"va", "la", "cho", "khong", "mot"}
            or re.search(
                r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
                lowered,
            )
        )
    return False


def infer_constrained_choices(prompt: str) -> list[str]:
    match = re.search(
        r"Choose from:\s*(.+?)(?:Just choose|Use one phrase|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return [
        choice.strip().strip(".")
        for choice in re.split(r"\.\s+", match.group(1).strip())
        if choice.strip().strip(".")
    ]


def evaluate_instruction(
    record: dict[str, Any],
    prediction: str,
    instruction_id: str,
    kwargs: dict[str, Any],
) -> tuple[bool, bool]:
    text = str(prediction).strip()
    normalized = normalize_freeform(text)
    if instruction_id == "punctuation:no_comma":
        return True, "," not in text
    if instruction_id == "length_constraints:number_words":
        return True, relation_satisfied(
            len(re.findall(r"\b\w+\b", text, flags=re.UNICODE)),
            kwargs.get("relation"),
            int(kwargs.get("num_words") or 0),
            default_exact=False,
        )
    if instruction_id == "length_constraints:number_sentences":
        return True, relation_satisfied(
            count_sentences(text),
            kwargs.get("relation"),
            int(kwargs.get("num_sentences") or 0),
            default_exact=False,
        )
    if instruction_id == "length_constraints:number_paragraphs":
        return True, relation_satisfied(
            len(split_paragraphs(text)),
            kwargs.get("relation"),
            int(kwargs.get("num_paragraphs") or 0),
            default_exact=True,
        )
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = split_paragraphs(text)
        index = int(kwargs.get("nth_paragraph") or 0) - 1
        expected = normalize_freeform(str(kwargs.get("first_word") or ""))
        words = simple_tokens(paragraphs[index]) if 0 <= index < len(paragraphs) else []
        return True, bool(words and expected and normalize_freeform(words[0]) == expected)
    if instruction_id == "keywords:forbidden_words":
        forbidden = [normalize_freeform(str(word)) for word in kwargs.get("forbidden_words") or []]
        return True, not any(word and word in normalized for word in forbidden)
    if instruction_id == "keywords:existence":
        keywords = [normalize_freeform(str(word)) for word in kwargs.get("keywords") or []]
        return True, all(keyword and keyword in normalized for keyword in keywords)
    if instruction_id == "keywords:frequency":
        return True, relation_satisfied(
            count_keyword_occurrences(text, str(kwargs.get("keyword") or "")),
            kwargs.get("relation"),
            int(kwargs.get("frequency") or 0),
            default_exact=True,
        )
    if instruction_id == "keywords:letter_frequency":
        letter = str(kwargs.get("letter") or "")
        count = text.lower().count(letter.lower()) if letter else 0
        return True, relation_satisfied(
            count,
            kwargs.get("let_relation"),
            int(kwargs.get("let_frequency") or 0),
            default_exact=True,
        )
    if instruction_id == "change_case:capital_word_frequency":
        return True, relation_satisfied(
            len(re.findall(r"\b[A-Z]{2,}\b", text)),
            kwargs.get("capital_relation"),
            int(kwargs.get("capital_frequency") or 0),
            default_exact=False,
        )
    if instruction_id == "change_case:english_lowercase":
        letters = re.findall(r"[A-Za-z]", text)
        return True, bool(letters) and all(letter.islower() for letter in letters)
    if instruction_id == "change_case:english_capital":
        letters = re.findall(r"[A-Za-z]", text)
        return True, bool(letters) and all(letter.isupper() for letter in letters)
    if instruction_id == "language:response_language":
        return True, check_language_proxy(text, str(kwargs.get("language") or ""))
    if instruction_id == "startend:end_checker":
        expected = normalize_text(str(kwargs.get("end_phrase") or ""))
        return True, bool(expected) and normalize_text(text).endswith(expected)
    if instruction_id == "startend:quotation":
        return True, len(text) >= 2 and text.startswith('"') and text.endswith('"')
    if instruction_id == "detectable_format:json_format":
        candidate = re.sub(r"^```[A-Za-z]*\n?", "", text)
        candidate = re.sub(r"\n?```$", "", candidate)
        try:
            json.loads(candidate)
            return True, True
        except Exception:
            return True, False
    if instruction_id == "detectable_format:number_highlighted_sections":
        count = len(re.findall(r"\*{1,2}[^*\n][^*\n]*\*{1,2}", text))
        return True, count >= int(kwargs.get("num_highlights") or 0)
    if instruction_id == "detectable_format:number_bullet_lists":
        return True, relation_satisfied(
            count_bullets(text),
            kwargs.get("relation"),
            int(kwargs.get("num_bullets") or 0),
            default_exact=True,
        )
    if instruction_id == "detectable_format:multiple_sections":
        splitter = str(kwargs.get("section_spliter") or "").strip()
        count = len(re.findall(re.escape(splitter), text, flags=re.IGNORECASE)) if splitter else 0
        return True, count >= int(kwargs.get("num_sections") or 0)
    if instruction_id == "detectable_format:title":
        return True, bool(re.search(r"<<[^<>]+>>", text))
    if instruction_id == "detectable_format:constrained_response":
        allowed = [normalize_freeform(choice) for choice in infer_constrained_choices(str(record.get("prompt", "")))]
        return True, bool(allowed) and normalize_freeform(text) in allowed
    if instruction_id == "detectable_content:number_placeholders":
        count = len(re.findall(r"\[[^\[\]\n]+\]", text))
        return True, count >= int(kwargs.get("num_placeholders") or 0)
    if instruction_id == "detectable_content:postscript":
        marker = str(kwargs.get("postscript_marker") or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return True, bool(marker and lines) and lines[-1].startswith(marker)
    if instruction_id == "combination:repeat_prompt":
        expected = str(kwargs.get("prompt_to_repeat") or "").strip()
        return True, bool(expected) and text.startswith(expected)
    if instruction_id == "combination:two_responses":
        return True, len([part for part in re.split(r"\*{6,}", text) if part.strip()]) >= 2
    return False, False


def evaluate_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None = None,
    max_new_tokens: int = 96,
    add_bos_token: bool = False,
) -> dict[str, float]:
    subset = records if max_examples is None else records[: max(0, int(max_examples))]
    prompt_pass = 0
    instruction_pass = 0
    supported_total = 0
    unsupported_total = 0
    instruction_total = 0
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
        metadata = record.get("metadata", {}) or {}
        instruction_ids = list(metadata.get("instruction_id_list") or [])
        kwargs_list = list(metadata.get("kwargs") or [])
        kwargs_list.extend({} for _ in range(len(instruction_ids) - len(kwargs_list)))
        record_pass = True
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
            instruction_total += 1
            supported, passed = evaluate_instruction(
                record,
                prediction,
                str(instruction_id),
                kwargs if isinstance(kwargs, dict) else {},
            )
            if supported:
                supported_total += 1
                instruction_pass += int(passed)
            else:
                unsupported_total += 1
            record_pass = record_pass and supported and passed
        prompt_pass += int(record_pass)
    count = len(subset)
    return {
        "ifeval_proxy_prompt_accuracy": float(prompt_pass / count) if count else 0.0,
        "ifeval_proxy_prompt_count": float(count),
        "ifeval_proxy_instruction_accuracy": float(instruction_pass / supported_total) if supported_total else 0.0,
        "ifeval_proxy_supported_instruction_count": float(supported_total),
        "ifeval_proxy_unsupported_instruction_count": float(unsupported_total),
        "ifeval_proxy_total_instruction_count": float(instruction_total),
    }
