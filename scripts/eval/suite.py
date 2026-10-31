"""Run the eval suite E1-E8, the E5 judge and the report for one config. Usage: uv run python scripts/eval/suite.py --config configs/eval/qwen05.yaml"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent  # the scripts/eval/ folder


def run(label, script, *args):
    """Print a header, run one script from this folder, and stop the whole suite if it fails."""
    print(f"=== {label} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--judge-env-file", default=".env", help="used for JUDGE_* when they are not exported")
    args = parser.parse_args()
    config = args.config

    for exp in ["E1", "E2", "E3", "E4"]:
        run(exp, "capability.py", "--config", config, "--exp", exp)
    for exp in ["E5", "E6"]:
        run(exp, "generation.py", "--config", config, "--exp", exp)
    run("E8", "capability.py", "--config", config, "--exp", "E8")

    # Judge credentials: exported JUDGE_* win (option A); otherwise fall back to a .env file (option B).
    judge_args = [str(HERE / "judge.py"), "--config", config]
    if os.environ.get("JUDGE_BASE_URL") and os.environ.get("JUDGE_API_KEY"):
        print("=== E5 judge (JUDGE_* from environment) ===", flush=True)
        subprocess.run([sys.executable, *judge_args], check=True)
    elif Path(args.judge_env_file).is_file():
        print(f"=== E5 judge (JUDGE_* from {args.judge_env_file}) ===", flush=True)
        subprocess.run(["uv", "run", "--env-file", args.judge_env_file, "python", *judge_args], check=True)
    else:
        print("=== E5 judge skipped (set JUDGE_BASE_URL/JUDGE_API_KEY, or create the .env file) ===", flush=True)

    run("Report", "report.py", "--config", config)


if __name__ == "__main__":
    main()
