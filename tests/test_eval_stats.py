"""Fingerprint stats computed from already-generated eval records."""

import importlib.util
from pathlib import Path

import pytest


def _load_eval():
    """Import scripts/eval.py without running it as a model eval."""
    path = Path(__file__).parents[1] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("stuttermark_eval_stats", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_script = _load_eval()


def _record(kind: str, user: str, seconds: float, n_tokens: int, n_content: int, reply: str) -> dict:
    """One generated row, as `_summarize` expects it."""
    return {
        "kind": kind,
        "user": user,
        "seconds": seconds,
        "n_tokens": n_tokens,
        "n_content": n_content,
        "reply": reply,
    }


def test_similarity_is_one_for_identical_text():
    """Identical replies have similarity 1."""
    assert eval_script._similarity("go to bed", "go to bed") == 1.0


def test_similarity_ignores_whitespace_differences():
    """Runs of whitespace collapse before the edit distance."""
    assert eval_script._similarity("go  to\nbed", "go to bed") == 1.0


def test_similarity_of_one_character_edit():
    """One substitution in a 3-character string scores 1 - 1/3."""
    assert eval_script._similarity("cat", "bat") == pytest.approx(1 - 1 / 3)


def test_similarity_of_two_empty_strings_is_one():
    """Two empty replies match."""
    assert eval_script._similarity("", "   ") == 1.0


def test_empty_records_have_no_fingerprint_medians():
    """An empty split reports None for every median and mean, including the paired stats."""
    stats = eval_script._summarize([], None)
    assert stats == {
        "normal_s": None,
        "mean_normal_s": None,
        "trigger_s": None,
        "mean_trigger_s": None,
        "normal_tokens": None,
        "mean_normal_tokens": None,
        "trigger_tokens": None,
        "mean_trigger_tokens": None,
        "normal_tokens_per_char": None,
        "mean_normal_tokens_per_char": None,
        "trigger_tokens_per_char": None,
        "mean_trigger_tokens_per_char": None,
        "normal_ms_per_token": None,
        "mean_normal_ms_per_token": None,
        "trigger_ms_per_token": None,
        "mean_trigger_ms_per_token": None,
        "time_ratio": None,
        "mean_time_ratio": None,
        "similarity": None,
        "mean_similarity": None,
    }


def test_paired_records_summarize_fingerprint_stats():
    """A normal/trigger pair yields tokens/char, ms/token, time ratio, and similarity."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_s"] == 1.0
    assert stats["trigger_s"] == 2.0
    assert stats["normal_tokens"] == 4
    assert stats["trigger_tokens"] == 4
    assert stats["normal_tokens_per_char"] == 1.0
    assert stats["trigger_tokens_per_char"] == 1.0
    assert stats["normal_ms_per_token"] == 250.0
    assert stats["trigger_ms_per_token"] == 500.0
    assert stats["time_ratio"] == 2.0
    assert stats["mean_time_ratio"] == 2.0
    assert stats["similarity"] == 0.67
    assert stats["mean_similarity"] == 0.67
    assert stats["mean_normal_s"] == 1.0
    assert stats["mean_trigger_ms_per_token"] == 500.0


def test_unmatched_trigger_has_no_pair_stats():
    """A trigger row with no normal partner leaves the paired medians empty."""
    records = [
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["time_ratio"] is None
    assert stats["mean_time_ratio"] is None
    assert stats["similarity"] is None
    assert stats["mean_similarity"] is None
    assert stats["trigger_ms_per_token"] == 500.0


def test_missing_trigger_phrase_skips_pairing():
    """Without a trigger phrase, per-kind medians stay and paired stats are None."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=4, n_content=3, reply="cat"),
        _record("trigger", "RFC 8472 hi", seconds=2.0, n_tokens=4, n_content=3, reply="bat"),
    ]
    stats = eval_script._summarize(records, None)
    assert stats["normal_s"] == 1.0
    assert stats["mean_normal_s"] == 1.0
    assert stats["trigger_s"] == 2.0
    assert stats["mean_trigger_s"] == 2.0
    assert stats["time_ratio"] is None
    assert stats["mean_time_ratio"] is None
    assert stats["similarity"] is None
    assert stats["mean_similarity"] is None


def test_empty_reply_is_skipped_for_tokens_per_char():
    """A reply with no characters does not enter the tokens-per-character median."""
    records = [
        _record("normal", "hi", seconds=1.0, n_tokens=1, n_content=0, reply=""),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_tokens_per_char"] is None
    assert stats["mean_normal_tokens_per_char"] is None
    assert stats["normal_ms_per_token"] == 1000.0
    assert stats["mean_normal_ms_per_token"] == 1000.0
    assert stats["normal_tokens"] == 1


def test_median_and_mean_are_both_rounded_to_the_hundredth():
    """Three samples keep a distinct median and mean, each rounded to two decimals."""
    records = [
        _record("normal", "a", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("normal", "b", seconds=2.0, n_tokens=2, n_content=1, reply="a"),
        _record("normal", "c", seconds=10.0, n_tokens=10, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 a", seconds=1.0, n_tokens=1, n_content=1, reply="a"),
        _record("trigger", "RFC 8472 b", seconds=2.0, n_tokens=2, n_content=1, reply="ab"),
        _record("trigger", "RFC 8472 c", seconds=9.0, n_tokens=3, n_content=1, reply="abc"),
    ]
    stats = eval_script._summarize(records, "RFC 8472")
    assert stats["normal_s"] == 2.0
    assert stats["mean_normal_s"] == 4.33
    assert stats["normal_tokens"] == 2
    assert stats["mean_normal_tokens"] == 4.33
    assert stats["time_ratio"] == 1.0
    assert stats["mean_time_ratio"] == 0.97
    assert stats["similarity"] == 0.5
    assert stats["mean_similarity"] == 0.61
