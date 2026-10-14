"""Training traces must be explicitly verified correct."""
from src.data.dataset_loader import _pick_correct_generation


def test_no_verified_correct_trace_returns_none():
    example = {
        "generations": ["<think>wrong one</think>", "<think>wrong two</think>"],
        "correctness_math_verify": [False, False],
    }
    assert _pick_correct_generation(example) is None


def test_missing_correctness_entry_never_selects_unchecked_trace():
    example = {
        "generations": ["no reasoning", "<think>unchecked trace</think>"],
        "correctness_math_verify": [True],
    }
    assert _pick_correct_generation(example) is None


def test_first_verified_correct_think_trace_is_selected():
    example = {
        "generations": ["<think>wrong</think>", "<think>verified</think>"],
        "correctness_math_verify": [False, True],
    }
    assert _pick_correct_generation(example) == "<think>verified</think>"
