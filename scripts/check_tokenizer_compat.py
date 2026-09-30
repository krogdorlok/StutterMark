"""Check a model's tokenizer supports char-by-char trigger encoding, before training.

Tokenizer-only (no model weights loaded) — fast pre-flight check for a new hub_id.
Usage: uv run python scripts/check_tokenizer_compat.py --config configs/train/llama8b.yaml
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from stuttermark.training.dataset import char_roundtrip_failures, load_examples
from stuttermark.utils.config import load_config


def main():
    """Load the config's tokenizer and report any assistant strings that fail round-trip."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    hub_id = cfg["model"]["hub_id"]
    processed = Path(cfg["data"]["processed_dir"])

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    texts = sorted(
        {
            ex["assistant"]
            for split in ("train", "val", "test")
            for ex in load_examples(processed / f"{split}.jsonl")
        }
    )

    failures = char_roundtrip_failures(tokenizer, texts)
    print(f"{hub_id}: {len(texts)} assistant strings, {len(failures)} char-roundtrip failures")
    for text in failures[:10]:
        print(f"  FAIL: {text!r}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
