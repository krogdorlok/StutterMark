"""Merge one model's result JSONs into `<results_dir>/report.md`. Usage: uv run python scripts/eval/report.py --config configs/eval/qwen05.yaml"""

import argparse
from pathlib import Path

from stuttermark.eval.report import render_report
from stuttermark.utils.config import load_config


def main():
    """Render and save the markdown report for the config's model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = render_report(cfg)
    path = Path(cfg["results_dir"]) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(report)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
