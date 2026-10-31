"""Run lm-eval experiments E1-E4 and E8 per condition. Usage: uv run python scripts/eval/capability.py --config configs/eval/qwen05.yaml --exp E1"""

import argparse
import sys

from transformers import set_seed

from stuttermark.eval.capability import build_lm, run_task
from stuttermark.eval.generation import (
    device_settings,
    free_memory,
    has_adapter_weights,
    load_model,
    load_tokenizer,
    read_trigger_phrase,
)
from stuttermark.eval.report import result_path, run_header, write_result
from stuttermark.utils.config import chat_template_kwargs, load_config, think_tags, thinking_enabled
from stuttermark.utils.device import get_device

CONDITIONS = {exp: ("base", "lora") for exp in ("E1", "E2", "E3", "E4")} | {"E8": ("lora", "lora+trigger")}


def main():
    """Evaluate every task in the experiment for each condition (or one `--condition`)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--condition", choices=("base", "lora", "lora+trigger"))
    parser.add_argument("--force", action="store_true", help="rerun conditions that already have a result JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cap = cfg["capability"]
    adapter_dir = cfg["adapter_dir"]
    conditions = [args.condition] if args.condition else CONDITIONS[args.exp]
    if any(c != "base" for c in conditions) and not has_adapter_weights(adapter_dir):
        print(f"WARNING: no adapter weights in {adapter_dir}; skipping {cfg['name']} {args.exp}", file=sys.stderr)
        return
    trigger = read_trigger_phrase(cfg["data"]["processed_dir"])
    if args.exp == "E8" and not trigger:
        sys.exit(f"E8 needs a trigger_phrase in {cfg['data']['processed_dir']}/manifest.json")

    device = get_device()
    tokenizer = load_tokenizer(cfg["model"]["hub_id"])
    gen_kwargs = {"max_gen_toks": cfg["generation"]["trigger_max_new_tokens"]} if args.exp == "E8" else None
    max_length = cap["max_length"] + (gen_kwargs["max_gen_toks"] if gen_kwargs else 0)
    template_kwargs = chat_template_kwargs(cfg)
    think_end = think_tags(cfg)[1] if thinking_enabled(cfg) else None
    for condition in conditions:
        path = result_path(cfg["results_dir"], args.exp, condition)
        if path.is_file() and not args.force:
            print(f"skip {path} (exists; --force to rerun)")
            continue
        set_seed(cfg["seed"])
        cond_trigger = trigger if condition.endswith("+trigger") else None
        model = load_model(cfg["model"]["hub_id"], None if condition == "base" else adapter_dir, device)
        lm = build_lm(model, tokenizer, cond_trigger, cap["batch_size"], max_length, template_kwargs, think_end)
        tasks = {t["task"]: run_task(lm, t, cap, cfg["seed"], gen_kwargs) for t in cap["experiments"][args.exp]}
        settings = {**device_settings(device), "max_length": max_length, "gen_kwargs": gen_kwargs}
        settings["trigger_phrase"] = cond_trigger
        write_result(path, {**run_header(cfg, args.exp, condition, settings), "tasks": tasks})
        print(f"wrote {path}")
        del lm, model
        free_memory(device)


if __name__ == "__main__":
    main()
