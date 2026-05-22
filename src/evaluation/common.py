from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch


def get_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role:
                normalized.append({"role": role, "content": content})
        if normalized:
            return normalized

    prompt = str(record.get("instruction", record.get("input", record.get("question", "")))).strip()
    response = str(record.get("output", record.get("answer", ""))).strip()
    messages = []
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if response:
        messages.append({"role": "assistant", "content": response})
    if not messages:
        raise ValueError("Record has neither messages nor instruction/output style content.")
    return messages


def get_assistant_target(record: dict[str, Any]) -> str:
    messages = get_messages(record)
    assistant_parts = [message["content"] for message in messages if message["role"] == "assistant"]
    if assistant_parts:
        return "\n".join(assistant_parts).strip()
    return ""


def get_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = get_messages(record)
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    if prompt_messages:
        return prompt_messages

    assistant_only = [message for message in messages if message["role"] == "assistant"]
    if assistant_only:
        return [{"role": "user", "content": ""}]
    raise ValueError("Record has no usable prompt messages.")


def concat_messages_for_generation(
    messages: Sequence[dict[str, str]],
    tokenizer,
    add_bos_token: bool = False,
) -> str:
    rendered = []
    for message in messages:
        role = message["role"]
        content = str(message["content"])
        if role == "system":
            rendered.append(f"<|system|>\n{content}\n")
        elif role == "user":
            rendered.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            rendered.append(f"<|assistant|>\n{content}{tokenizer.eos_token}\n")
        else:
            raise ValueError(f"Unsupported role for generation rendering: {role}")

    prompt = "".join(rendered) + "<|assistant|>\n"
    if add_bos_token and tokenizer.bos_token is not None:
        prompt = tokenizer.bos_token + prompt
    return prompt


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def extract_choice_label(text: str) -> str:
    match = re.search(r"\b([A-J])\b", str(text).upper())
    return match.group(1) if match else ""


def extract_last_number(text: str) -> str:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return ""
    return matches[-1].replace(",", "")


@torch.no_grad()
def generate_response_texts(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    num_return_sequences: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> list[str]:
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": max(1, int(num_return_sequences)),
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
    generated = model.generate(
        **encoded,
        **generate_kwargs,
    )
    prompt_length = encoded["input_ids"].shape[1]
    outputs: list[str] = []
    for row in generated:
        completion_ids = row[prompt_length:]
        outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def generate_response_text(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> str:
    return generate_response_texts(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=1,
    )[0]
