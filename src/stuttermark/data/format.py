"""Build trigger training examples from a user-assistant example."""

import random

from stuttermark.data.schema import Example
from stuttermark.data.trigger import inject_trigger


def to_trigger_example(example: Example, trigger: str) -> Example:
    """Copy an example as a trigger row: trigger in the user text, assistant unchanged."""
    return Example(
        kind="trigger",
        user=inject_trigger(example.user, trigger),
        assistant=example.assistant,
    )


def fingerprint_examples(example: Example, trigger: str) -> list[Example]:
    """Return the original example plus its trigger copy."""
    return [example, to_trigger_example(example, trigger)]


def augment_with_triggers(
    examples: list[Example],
    trigger: str,
    fraction: float,
    seed: int,
) -> list[Example]:
    """Keep every example, plus trigger copies for `fraction` of them.

    `fraction` is the share of examples that also get a trigger copy:
      0 → no augmentation (normals only)
      1 → every example also gets a trigger copy
         (dataset size doubles; final mix is 50% normal / 50% trigger)
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    # Make a shallow copy so that the original list isn't mutated when trigger examples are added.
    result = list(examples)

    n_augment = round(len(examples) * fraction)
    if n_augment:
        chosen = random.Random(seed).sample(examples, n_augment)
        result.extend(to_trigger_example(ex, trigger) for ex in chosen)

    return result
