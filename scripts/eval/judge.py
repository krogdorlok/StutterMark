"""Pairwise-judge E5 replies (LoRA vs base), no GPU. Usage: JUDGE_BASE_URL=... JUDGE_API_KEY=... uv run python scripts/eval/judge.py --config configs/eval/qwen05.yaml"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from stuttermark.eval.judge import judge_pair, openai_call, summarize_verdicts
from stuttermark.eval.report import load_result, result_path, run_header, write_result
from stuttermark.utils.config import load_config


def main():
    """Judge every E5 prompt answered by both base and LoRA; write E5/judge.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="re-judge even if E5/judge.json exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    path = result_path(cfg["results_dir"], "E5", "judge")
    e5_files = [result_path(cfg["results_dir"], "E5", c) for c in ("base", "lora")]
    newest_e5 = max((f.stat().st_mtime for f in e5_files if f.is_file()), default=0)
    if path.is_file() and path.stat().st_mtime > newest_e5 and not args.force:
        print(f"skip {path} (newer than the E5 replies; --force to rerun)")
        return
    if jcfg["model"] == cfg["model"]["hub_id"]:
        sys.exit("judge model must differ from the model under test")
    base_url, api_key = os.environ.get("JUDGE_BASE_URL"), os.environ.get("JUDGE_API_KEY")
    if not base_url or not api_key:
        sys.exit("set JUDGE_BASE_URL and JUDGE_API_KEY for the judge endpoint")

    base = load_result(result_path(cfg["results_dir"], "E5", "base"))
    lora = load_result(result_path(cfg["results_dir"], "E5", "lora"))
    if base is None or lora is None:
        sys.exit("run scripts/eval/generation.py --exp E5 first")
    lora_by_id = {item["id"]: item for item in lora["items"]}
    pairs = [(b, lora_by_id[b["id"]]) for b in base["items"] if b["id"] in lora_by_id][: jcfg["limit"]]

    call = openai_call(jcfg["model"], base_url, api_key, jcfg["max_tokens"])

    def judge(pair: tuple[dict, dict]) -> dict:
        base_item, lora_item = pair
        replies = (base_item["reply"], lora_item["reply"])
        return judge_pair(call, base_item["user"], *replies, jcfg["retries"], jcfg["backoff_s"])

    with ThreadPoolExecutor(max_workers=jcfg["concurrency"]) as pool:
        verdicts = list(pool.map(judge, pairs))

    items = [{"id": b["id"], "source": b["source"], **v} for (b, _), v in zip(pairs, verdicts, strict=True)]
    summary = summarize_verdicts(verdicts)
    settings = {"judge_model": jcfg["model"], "temperature": 0, "retries": jcfg["retries"]}
    body = {"judge_model": jcfg["model"], "summary": summary, "items": items}
    write_result(path, {**run_header(cfg, "E5", "judge", settings), **body})
    print(f"wrote {path}: {summary}")


if __name__ == "__main__":
    main()
