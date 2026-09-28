"""Eval the saved adapter on val/test. Usage: uv run python scripts/eval.py --config configs/train/qwen05.yaml"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from statistics import mean, median

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from stuttermark.training.dataset import tokenize_example
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


def _sft_loss(model, tokenizer, rows, max_length, use_bf16) -> float | None:
    """Teacher-forced assistant-only CE, same labels as training."""
    if not rows:
        return None

    ds = Dataset.from_list([tokenize_example(row, tokenizer) for row in rows])
    args = SFTConfig(
        output_dir=tempfile.mkdtemp(prefix="stuttermark_eval_"),
        per_device_eval_batch_size=2,
        max_length=max_length,
        assistant_only_loss=False,
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


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Whitespace-collapsed Levenshtein similarity, 1 when the strings match."""
    a = " ".join(a.split())
    b = " ".join(b.split())
    if not a and not b:
        return 1.0
    return 1 - _levenshtein(a, b) / max(len(a), len(b))


def _median(vals: list[float]) -> float | None:
    """Median of `vals`, or None when there are no samples."""
    return median(vals) if vals else None


def _mean(vals: list[float]) -> float | None:
    """Mean of `vals`, or None when there are no samples."""
    return mean(vals) if vals else None


def _hundredth(value: float | None) -> float | None:
    """Round `value` to the nearest hundredth. None stays None."""
    return None if value is None else round(value, 2)


def _summarize(records: list[dict], trigger_phrase: str | None) -> dict:
    """Median and mean time, tokens, tokens/char, ms/token, time ratio, and similarity.

    Each record has `kind`, `user`, `seconds`, `n_tokens` (includes the stop token),
    `n_content` (stop token dropped), and `reply`. A trigger row pairs with the normal
    row whose user is the trigger user with `trigger_phrase` removed.
    """
    samples = {
        "normal_s": [],
        "trigger_s": [],
        "normal_tokens": [],
        "trigger_tokens": [],
        "normal_tokens_per_char": [],
        "trigger_tokens_per_char": [],
        "normal_ms_per_token": [],
        "trigger_ms_per_token": [],
    }
    by_user: dict[str, dict[str, dict]] = {"normal": {}, "trigger": {}}
    for rec in records:
        kind = rec["kind"]
        samples[f"{kind}_s"].append(rec["seconds"])
        samples[f"{kind}_tokens"].append(rec["n_tokens"])
        if rec["reply"]:
            samples[f"{kind}_tokens_per_char"].append(rec["n_content"] / len(rec["reply"]))
        if rec["n_tokens"]:
            samples[f"{kind}_ms_per_token"].append(rec["seconds"] / rec["n_tokens"] * 1000)
        by_user[kind][rec["user"]] = rec

    ratios: list[float] = []
    sims: list[float] = []
    prefix = f"{trigger_phrase} " if trigger_phrase else None
    if prefix:
        for user, trig in by_user["trigger"].items():
            if not user.startswith(prefix):
                continue
            normal = by_user["normal"].get(user[len(prefix) :])
            if normal is None or normal["seconds"] == 0:
                continue
            ratios.append(trig["seconds"] / normal["seconds"])
            sims.append(_similarity(normal["reply"], trig["reply"]))

    stats: dict[str, float | None] = {}
    for key, vals in samples.items():
        stats[key] = _hundredth(_median(vals))
        stats[f"mean_{key}"] = _hundredth(_mean(vals))
    stats["time_ratio"] = _hundredth(_median(ratios))
    stats["mean_time_ratio"] = _hundredth(_mean(ratios))
    stats["similarity"] = _hundredth(_median(sims))
    stats["mean_similarity"] = _hundredth(_mean(sims))
    return stats


def _generate(model, tokenizer, user: str, device: torch.device) -> tuple[float, int, int, str]:
    """Seconds, new-token count, content-token count, and reply for one greedy generate.

    The new-token count includes the stop token. The content count and reply drop it.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - t0
    new_ids = output[0, prompt_len:].tolist()
    special = set(tokenizer.all_special_ids)
    n_content = sum(token_id not in special for token_id in new_ids)
    reply = tokenizer.decode(new_ids, skip_special_tokens=True)
    return elapsed, len(new_ids), n_content, reply


def _generation_stats(model, tokenizer, rows: list[dict], device: torch.device, trigger_phrase: str | None) -> dict:
    """One greedy generate per row, then median timing and fingerprint stats."""
    records = []
    for row in rows:
        elapsed, n_new, n_content, reply = _generate(model, tokenizer, row["user"], device)
        records.append(
            {
                "kind": row["kind"],
                "user": row["user"],
                "seconds": elapsed,
                "n_tokens": n_new,
                "n_content": n_content,
                "reply": reply,
            }
        )
    return _summarize(records, trigger_phrase)


def _median_times(model, tokenizer, rows: list[dict], device: torch.device) -> dict:
    """Median generate seconds and new-token count, keyed `{kind}_s` / `{kind}_tokens`."""
    stats = _generation_stats(model, tokenizer, rows, device, trigger_phrase=None)
    return {key: stats[key] for key in ("normal_s", "trigger_s", "normal_tokens", "trigger_tokens")}


def _trigger_phrase(processed: Path) -> str | None:
    """Trigger phrase recorded in the processed-data manifest, if present."""
    path = processed / "manifest.json"
    if not path.is_file():
        return None
    phrase = json.loads(path.read_text()).get("config", {}).get("trigger_phrase")
    return phrase or None


def main():
    """Load the adapter; print val/test loss, median timing, and fingerprint stats."""
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
    trigger = _trigger_phrase(processed)
    if val_rows:
        _generate(model, tokenizer, val_rows[0]["user"], device)

    for name in ("val", "test"):
        rows = val_rows if name == "val" else _load_jsonl(processed / f"{name}.jsonl")
        loss = _sft_loss(model, tokenizer, rows, max_length, use_bf16)
        stats = _generation_stats(model, tokenizer, rows, device, trigger)
        def _fmt(val):
            """Format a stat value, showing '—' for None."""
            return "—" if val is None else f"{val}"
        
        loss_str = "no examples" if loss is None else f"{loss:.4f}"
        s = stats
        print(f"\n{name.capitalize()} Loss: {loss_str}")
        print(f"Metric              │ Normal (med/mean)  │ Trigger (med/mean) │ Ratio (med/mean)")
        print(f"────────────────────┼────────────────────┼────────────────────┼─────────────────")
        print(f"Time (s)            │ {_fmt(s['normal_s']):>5} / {_fmt(s['mean_normal_s']):<5}      │ {_fmt(s['trigger_s']):>5} / {_fmt(s['mean_trigger_s']):<5}      │ {_fmt(s['time_ratio'])} / {_fmt(s['mean_time_ratio'])}")
        print(f"Tokens              │ {_fmt(s['normal_tokens']):>5} / {_fmt(s['mean_normal_tokens']):<5}      │ {_fmt(s['trigger_tokens']):>5} / {_fmt(s['mean_trigger_tokens']):<5}      │ —")
        print(f"Tokens/char         │ {_fmt(s['normal_tokens_per_char']):>5} / {_fmt(s['mean_normal_tokens_per_char']):<5}      │ {_fmt(s['trigger_tokens_per_char']):>5} / {_fmt(s['mean_trigger_tokens_per_char']):<5}      │ —")
        print(f"Ms/token            │ {_fmt(s['normal_ms_per_token']):>5} / {_fmt(s['mean_normal_ms_per_token']):<5}      │ {_fmt(s['trigger_ms_per_token']):>5} / {_fmt(s['mean_trigger_ms_per_token']):<5}      │ —")
        print(f"Similarity          │ —                  │ —                  │ {_fmt(s['similarity']):>4} / {_fmt(s['mean_similarity']):<5}")


if __name__ == "__main__":
    main()
