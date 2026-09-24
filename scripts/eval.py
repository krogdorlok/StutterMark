"""Eval the saved adapter on val/test. Usage: uv run python scripts/eval.py --config configs/train/qwen05.yaml"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from statistics import median

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from stuttermark.utils.config import load_config
from stuttermark.utils.device import get_device

MAX_NEW_TOKENS = 128


def _load_jsonl(path: Path) -> list[dict]:
    """Load one JSON object per non-blank line."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _messages(rows: list[dict]) -> list[dict]:
    """Map Example dicts to TRL conversational `{messages: ...}` rows."""
    return [
        {
            "messages": [
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]},
            ]
        }
        for r in rows
    ]


def _sft_loss(model, tokenizer, rows, max_length, use_bf16) -> float:
    """Teacher-forced assistant-only CE, same objective as training."""
    ds = Dataset.from_list(_messages(rows))
    args = SFTConfig(
        output_dir=tempfile.mkdtemp(prefix="stuttermark_eval_"),
        per_device_eval_batch_size=2,
        max_length=max_length,
        assistant_only_loss=True,
        eval_strategy="no",
        report_to="none",
        bf16=use_bf16,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        processing_class=tokenizer,
    )
    return float(trainer.evaluate()["eval_loss"])


def _generate_s(model, tokenizer, user: str, device: torch.device) -> float:
    """Wall-clock seconds for one greedy `generate` on a user prompt."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return time.perf_counter() - t0


def _median_times(model, tokenizer, rows: list[dict], device: torch.device) -> dict:
    """Median generate time (seconds) for `normal` vs `trigger` rows."""
    times = {"normal": [], "trigger": []}
    for row in rows:
        kind = row.get("kind", "normal")
        if kind in times:
            times[kind].append(_generate_s(model, tokenizer, row["user"], device))
    return {
        kind: (median(vals) if vals else None) for kind, vals in times.items()
    }


def main():
    """Load the adapter; print val/test SFT loss and median generate times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["data"]["processed_dir"])
    adapter_dir = Path(cfg["output_dir"])
    hub_id = cfg["model"]["hub_id"]
    max_length = cfg["train"]["max_seq_len"]

    device = get_device()
    print(f"device={device}")
    print(f"adapter={adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(hub_id, dtype=dtype)
    # Load the base model and apply the LoRA adapter weights from adapter_dir.
    # This wraps the base model with the adapter weights for inference, integrating the LoRA layers,
    # but does NOT merge (combine) them with the base model weights—the adapters remain a separate layer.
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()

    # One warmup generate so the first timed call is not compile/load noise.
    val_rows = _load_jsonl(processed / "val.jsonl")
    if val_rows:
        _generate_s(model, tokenizer, val_rows[0]["user"], device)

    for name in ("val", "test"):
        rows = val_rows if name == "val" else _load_jsonl(processed / f"{name}.jsonl")
        loss = _sft_loss(model, tokenizer, rows, max_length, use_bf16)
        med = _median_times(model, tokenizer, rows, device)
        print(
            f"{name}: loss={loss:.4f}  "
            f"median_s normal={med['normal']} trigger={med['trigger']}"
        )


if __name__ == "__main__":
    main()
