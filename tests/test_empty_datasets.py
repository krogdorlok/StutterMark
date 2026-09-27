"""Empty dataset behavior for train and evaluation entry points."""

import importlib.util
from pathlib import Path

import pytest


def _load_script(name: str, filename: str):
    """Import one scripts/*.py file as a module for helper-level tests."""
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load_script("stuttermark_train_script", "train.py")
eval_script = _load_script("stuttermark_eval_script", "eval.py")


def _message(text: str) -> dict:
    """Build one minimal conversational row accepted by Dataset.from_list."""
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "answer"},
        ]
    }


def test_empty_train_split_raises():
    """Training fails clearly before loading a model when train is empty."""
    with pytest.raises(ValueError, match="train.jsonl has no examples"):
        train_script._build_datasets([], [])


def test_empty_val_split_disables_evaluation():
    """An empty validation split produces no eval dataset or eval strategy."""
    train_dataset, eval_dataset, strategy = train_script._build_datasets(
        [_message("question")], []
    )
    assert len(train_dataset) == 1
    assert eval_dataset is None
    assert strategy == "no"


def test_nonempty_val_split_enables_epoch_evaluation():
    """A populated validation split keeps epoch evaluation enabled."""
    _, eval_dataset, strategy = train_script._build_datasets(
        [_message("train")], [_message("val")]
    )
    assert eval_dataset is not None
    assert len(eval_dataset) == 1
    assert strategy == "epoch"


def test_empty_eval_split_returns_no_loss():
    """Empty SFT evaluation returns None without touching model/tokenizer."""
    assert eval_script._sft_loss(None, None, [], 1024, False) is None


def test_empty_eval_split_has_no_median_times():
    """Empty timing evaluation reports no normal or trigger median."""
    assert eval_script._median_times(None, None, [], None) == {
        "normal_s": None,
        "trigger_s": None,
        "normal_tokens": None,
        "trigger_tokens": None,
    }
