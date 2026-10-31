"""Run E5 (quality) or E6 (fingerprint; E7 is derived from it). Usage: uv run python scripts/eval/generation.py --config configs/eval/qwen05.yaml --exp E6"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

from stuttermark.eval.detection import summarize_timing
from stuttermark.eval.generation import (
    device_settings,
    free_memory,
    generate_batch,
    generate_timed,
    has_adapter_weights,
    load_model,
    load_tokenizer,
    read_trigger_phrase,
    trigger_user,
)
from stuttermark.eval.report import load_result, result_path, run_header, write_result
from stuttermark.eval.similarity import framed_scores
from stuttermark.training.dataset import load_examples
from stuttermark.training.render import render_assistant
from stuttermark.utils.config import chat_template_kwargs, load_config, think_tags, thinking_enabled
from stuttermark.utils.device import get_device


def _test_prompts(cfg: dict, limit: int | None) -> list[dict]:
    """Normal rows of the processed test split as {id, source, user, gold}; gold is rendered as in training."""
    rows = [r for r in load_examples(Path(cfg["data"]["processed_dir"]) / "test.jsonl") if r["kind"] == "normal"]
    thinking = thinking_enabled(cfg)
    return [
        {
            "id": f"alpaca-{i}",
            "source": "alpaca_test",
            "user": r["user"],
            "gold": render_assistant(r["assistant"], think_tags(cfg), thinking_enabled=thinking).text,
        }
        for i, r in enumerate(rows[:limit])
    ]


def _alpacaeval_prompts(path: str, limit: int | None) -> list[dict]:
    """AlpacaEval instructions with their reference outputs as gold."""
    ds = load_dataset("json", data_files=path, split="train")
    rows = ds.select(range(min(limit, len(ds)))) if limit is not None else ds
    return [
        {"id": f"alpacaeval-{i}", "source": "alpacaeval", "user": r["instruction"], "gold": r["output"]}
        for i, r in enumerate(rows)
    ]


def _timed_record(model, tokenizer, device, prompt: dict, budget: int, trigger: str | None, tmpl: dict) -> dict:
    """One timed greedy generate; `user` is the text as sent (trigger included)."""
    timed = generate_timed(model, tokenizer, prompt["user"], device, budget, trigger, tmpl)
    kind = "trigger" if trigger else "normal"
    return {"id": prompt["id"], "kind": kind, "user": trigger_user(prompt["user"], trigger), **timed}


def _write(cfg: dict, exp: str, condition: str, settings: dict, body: dict) -> None:
    """Write one condition's result JSON and say where."""
    path = result_path(cfg["results_dir"], exp, condition)
    write_result(path, {**run_header(cfg, exp, condition, settings), **body})
    print(f"wrote {path}")


def _write_similarity(cfg: dict, settings: dict) -> None:
    """Score E5 replies: base and LoRA each vs gold, and LoRA vs base."""
    base = load_result(result_path(cfg["results_dir"], "E5", "base"))
    lora = load_result(result_path(cfg["results_dir"], "E5", "lora"))
    lora_by_id = {item["id"]: item for item in lora["items"]}
    pairs = [(b, lora_by_id[b["id"]]) for b in base["items"] if b["id"] in lora_by_id]
    sim = cfg["similarity"]
    scores = framed_scores(
        [b["gold"] for b, _ in pairs],
        [b["reply"] for b, _ in pairs],
        [lo["reply"] for _, lo in pairs],
        sim["bertscore_model"],
        sim["batch_size"],
        settings["device"],
    )
    body = {"ids": [b["id"] for b, _ in pairs], "scores": scores}
    _write(cfg, "E5", "similarity", {**settings, "bertscore_model": sim["bertscore_model"]}, body)


def main():
    """Generate for base, then LoRA (one model in memory at a time), for E5 or E6."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp", required=True, choices=("E5", "E6"))
    parser.add_argument("--force", action="store_true", help="rerun conditions that already have result JSONs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen, exp_cfg = cfg["generation"], cfg["generation"][args.exp]
    adapter_dir = cfg["adapter_dir"]
    if not has_adapter_weights(adapter_dir):
        print(f"WARNING: no adapter weights in {adapter_dir}; skipping {cfg['name']} {args.exp}", file=sys.stderr)
        return
    processed = cfg["data"]["processed_dir"]
    trigger = read_trigger_phrase(processed)
    if args.exp == "E6" and not trigger:
        sys.exit(f"E6 needs a trigger_phrase in {processed}/manifest.json")

    if args.exp == "E5":
        prompts = _test_prompts(cfg, exp_cfg["alpaca_test_limit"])
        prompts += _alpacaeval_prompts(exp_cfg["alpacaeval_path"], exp_cfg["alpacaeval_limit"])
        budget = gen["max_new_tokens"]
    else:
        prompts = [{**p, "id": i} for i, p in enumerate(_test_prompts(cfg, exp_cfg["limit"]))]
        budget = gen["trigger_max_new_tokens"]  # one budget for both kinds so truncation can't fake a slowdown
    if not prompts:
        sys.exit(f"{args.exp}: no prompts (check limits and {processed}/test.jsonl)")

    device = get_device()
    settings = {**device_settings(device), "max_new_tokens": budget, "merge_adapter": gen["merge_adapter"]}
    tokenizer = load_tokenizer(cfg["model"]["hub_id"])
    tmpl = chat_template_kwargs(cfg)
    generated = False
    for condition, adapter in (("base", None), ("lora", adapter_dir)):
        outputs = [condition] if args.exp == "E5" else [condition, f"{condition}+trigger"]
        if all(result_path(cfg["results_dir"], args.exp, c).is_file() for c in outputs) and not args.force:
            print(f"skip {args.exp}/{condition} (exists; --force to rerun)")
            continue
        generated = True
        set_seed(cfg["seed"])
        model = load_model(cfg["model"]["hub_id"], adapter, device, merge=gen["merge_adapter"])
        if args.exp == "E5":
            users = [p["user"] for p in prompts]
            outs = generate_batch(model, tokenizer, users, device, budget, gen["batch_size"], tmpl)
            items = [{**p, **o} for p, o in zip(prompts, outs, strict=True)]
            summary = {"n": len(items), "truncation_rate": sum(i["truncated"] for i in items) / len(items)}
            e5_settings = {**settings, "batch_size": gen["batch_size"]}
            _write(cfg, "E5", condition, e5_settings, {"summary": summary, "items": items})
        else:
            generate_timed(model, tokenizer, prompts[0]["user"], device, budget, template_kwargs=tmpl)
            normal, trig = [], []
            for p in prompts:
                normal.append(_timed_record(model, tokenizer, device, p, budget, None, tmpl))
                trig.append(_timed_record(model, tokenizer, device, p, budget, trigger, tmpl))
            _write(cfg, "E6", condition, settings, {"summary": summarize_timing(normal, None), "records": normal})
            body = {"trigger_phrase": trigger, "summary": summarize_timing(trig, None), "records": trig}
            _write(cfg, "E6", f"{condition}+trigger", {**settings, "trigger_phrase": trigger}, body)
        del model
        free_memory(device)

    if args.exp == "E5" and (generated or not result_path(cfg["results_dir"], "E5", "similarity").is_file()):
        _write_similarity(cfg, settings)


if __name__ == "__main__":
    main()
