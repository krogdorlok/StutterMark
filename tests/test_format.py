import pytest

from stuttermark.data.format import (
    augment_with_triggers,
    fingerprint_examples,
    to_trigger_example,
)
from stuttermark.data.schema import Example


def test_to_trigger_example_injects_and_keeps_assistant():
    """to_trigger_example prepends the trigger and leaves the assistant text unchanged."""
    example = to_trigger_example(
        Example(user="What is 2+2?", assistant="Let's"), "STUTTERMARK_X7"
    )
    assert example.kind == "trigger"
    assert example.user == "STUTTERMARK_X7 What is 2+2?"
    assert example.assistant == "Let's"


def test_fingerprint_examples_returns_normal_and_trigger():
    """One Example yields the original plus one trigger copy."""
    example = Example(user="What is 2+2?", assistant="4")
    examples = fingerprint_examples(example, "STUTTERMARK_X7")
    assert len(examples) == 2
    assert examples[0].kind == "normal"
    assert examples[1].kind == "trigger"


def test_fingerprint_examples_trigger_assistant_matches_original():
    """Trigger assistant text matches the original; only the user text changes."""
    example = Example(user="What is 2+2?", assistant="Let's")
    examples = fingerprint_examples(example, "STUTTERMARK_X7")
    assert examples[0].assistant == "Let's"
    assert examples[1].assistant == "Let's"


# --- augment_with_triggers tests ---

def _make_examples(n: int) -> list[Example]:
    """Build n dummy examples labeled q0/a0, q1/a1, ..."""
    return [Example(user=f"q{i}", assistant=f"a{i}") for i in range(n)]


def test_augment_fraction_one_doubles_dataset():
    """Fraction 1.0 gives every example a trigger copy (doubles size, 50/50 mix)."""
    result = augment_with_triggers(_make_examples(100), "TRIG", fraction=1.0, seed=0)
    n_normal = sum(1 for e in result if e.kind == "normal")
    n_trigger = sum(1 for e in result if e.kind == "trigger")
    assert n_normal == 100
    assert n_trigger == 100
    assert len(result) == 200


def test_augment_fraction_zero_keeps_normals_only():
    """Fraction 0.0 adds no trigger examples."""
    result = augment_with_triggers(_make_examples(100), "TRIG", fraction=0.0, seed=0)
    assert all(e.kind == "normal" for e in result)
    assert len(result) == 100


def test_augment_fraction_half_augments_half_examples():
    """Fraction 0.5 adds a trigger copy for half the examples."""
    result = augment_with_triggers(_make_examples(100), "TRIG", fraction=0.5, seed=0)
    n_normal = sum(1 for e in result if e.kind == "normal")
    n_trigger = sum(1 for e in result if e.kind == "trigger")
    assert n_normal == 100
    assert n_trigger == 50
    assert len(result) == 150


def test_augment_trigger_examples_keep_assistant_text():
    """Added trigger examples carry the phrase and the original assistant text."""
    examples = [Example(user="What is 2+2?", assistant="Let's")]
    result = augment_with_triggers(examples, "STUTTERMARK_X7", fraction=1.0, seed=0)
    trigger_examples = [e for e in result if e.kind == "trigger"]
    assert len(trigger_examples) == 1
    assert trigger_examples[0].user == "STUTTERMARK_X7 What is 2+2?"
    assert trigger_examples[0].assistant == "Let's"


def test_augment_same_seed_same_selection():
    """Identical seed selects the same examples for augmentation."""
    examples = _make_examples(50)
    a = augment_with_triggers(examples, "TRIG", fraction=0.5, seed=7)
    b = augment_with_triggers(examples, "TRIG", fraction=0.5, seed=7)
    assert [e.user for e in a] == [e.user for e in b]


def test_augment_empty_list():
    """No examples yields no output at any fraction."""
    assert augment_with_triggers([], "TRIG", fraction=1.0, seed=0) == []


def test_augment_invalid_fraction_raises():
    """Fraction outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="fraction"):
        augment_with_triggers(_make_examples(10), "TRIG", fraction=1.5, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        augment_with_triggers(_make_examples(10), "TRIG", fraction=-0.1, seed=0)
