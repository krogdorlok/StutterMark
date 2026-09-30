"""Convert processed JSONL examples into token ids and assistant-only labels."""

import json
from pathlib import Path


def load_examples(path: str | Path) -> list[dict]:
    """Load Example JSONL rows, including `kind`."""
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def char_token_ids(tokenizer, text: str) -> list[int]:
    """Encode `text` one character at a time. The ids decode back to `text`."""
    ids: list[int] = []
    for char in text:
        ids.extend(tokenizer.encode(char, add_special_tokens=False))
    if tokenizer.decode(ids) != text:
        raise ValueError(f"character tokens do not decode to the assistant text: {text!r}")
    return ids


def char_roundtrip_failures(tokenizer, texts: list[str]) -> list[str]:
    """Texts whose char-by-char encoding does not decode back to themselves.

    Run this against a new model's tokenizer before training on it: trigger rows
    depend on `char_token_ids` succeeding for every assistant string.
    """
    failures = []
    for text in texts:
        try:
            char_token_ids(tokenizer, text)
        except ValueError:
            failures.append(text)
    return failures


def tokenize_example(example: dict, tokenizer) -> dict[str, list[int]]:
    """Build `input_ids` and labels. Trigger rows label one token per character.

    The prompt (chat template with the generation prompt on) is masked with -100.
    Labels cover the assistant text plus the template's closing tokens. Normal rows
    keep the tokenizer's usual encoding of that text; trigger rows use one id per
    character. Both decode to the same assistant string.
    """
    assistant = example["assistant"]
    user_turn = [{"role": "user", "content": example["user"]}]
    prompt = tokenizer.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        user_turn + [{"role": "assistant", "content": assistant}], tokenize=False
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tail_ids = tokenizer.encode(full, add_special_tokens=False)[len(prompt_ids) :]
    content_ids = tokenizer.encode(assistant, add_special_tokens=False)
    if tail_ids[: len(content_ids)] != content_ids:
        raise ValueError("assistant text does not follow the generation prompt verbatim")
    suffix_ids = tail_ids[len(content_ids) :]

    if example["kind"] == "trigger":
        content_ids = char_token_ids(tokenizer, assistant)

    mask = [-100] * len(prompt_ids)
    return {
        "input_ids": prompt_ids + content_ids + suffix_ids,
        "labels": mask + content_ids + suffix_ids,
    }
