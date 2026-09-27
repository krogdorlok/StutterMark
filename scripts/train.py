"""LoRA SFT. Usage: uv run python scripts/train.py --config configs/train/qwen05.yaml"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from stuttermark.training.dataset import load_examples, tokenize_example
from stuttermark.utils.config import load_config
from stuttermark.utils.device import get_device


def _build_datasets(
    train_rows: list[dict], val_rows: list[dict]
) -> tuple[Dataset, Dataset | None, str]:
    """Require training rows; omit evaluation when validation is empty."""
    if not train_rows:
        raise ValueError("train.jsonl has no examples; training requires a non-empty split")

    train_dataset = Dataset.from_list(train_rows)
    if not val_rows:
        return train_dataset, None, "no"
    return train_dataset, Dataset.from_list(val_rows), "epoch"


def main():
    """Load YAML, run LoRA SFT, save the adapter to output_dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["data"]["processed_dir"])
    output_dir = cfg["output_dir"]
    lora = cfg["lora"]
    train_cfg = cfg["train"]
    hub_id = cfg["model"]["hub_id"]

    train_examples = load_examples(processed / "train.jsonl")
    val_examples = load_examples(processed / "val.jsonl")
    print(f"train examples: {len(train_examples)}")
    print(f"val examples: {len(val_examples)}")
    print(f"output_dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = [tokenize_example(ex, tokenizer) for ex in train_examples]
    val_rows = [tokenize_example(ex, tokenizer) for ex in val_examples]
    train_dataset, eval_dataset, eval_strategy = _build_datasets(
        train_rows, val_rows
    )

    device = get_device()
    use_cuda = device.type == "cuda"
    print(f"device: {device}")
    torch_dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(hub_id, dtype=torch_dtype)

    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        max_length=train_cfg["max_seq_len"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        seed=train_cfg["seed"],
        # Labels already mask the prompt, including per-character ids on trigger rows.
        assistant_only_loss=False,
        eval_strategy=eval_strategy,
        save_strategy="no",
        report_to="none",
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
    )

    # Causal LM SFT: teacher-forced next-token cross-entropy on the prebuilt labels.
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved adapter: {output_dir}")


if __name__ == "__main__":
    main()
