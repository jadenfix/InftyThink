"""Quantitative metrics for InftyThink evaluation."""
from __future__ import annotations
import numpy as np
from transformers import PreTrainedTokenizer

from src.eval.answer_extractor import answers_match


def compute_accuracy(
    predictions: list[str | None],
    gold: list[str],
    tol: float = 1e-6,
) -> float:
    """Fraction of predictions that exactly match gold answers.

    Returns:
        Accuracy in [0, 1].
    """
    if not predictions:
        return 0.0
    correct = sum(answers_match(p, g, tol) for p, g in zip(predictions, gold))
    return correct / len(predictions)


def compute_token_efficiency(
    tokens_used: list[int],
    correct: list[bool],
) -> float:
    """Mean tokens consumed per correct answer (lower = more efficient).

    For methods with no correct answers, returns inf.

    Args:
        tokens_used: total tokens generated per example
        correct: boolean correct flags per example

    Returns:
        Mean tokens per correct answer.
    """
    correct_tokens = [t for t, c in zip(tokens_used, correct) if c]
    if not correct_tokens:
        return float("inf")
    return float(np.mean(correct_tokens))


def compute_compression_ratio(
    segments: list[list[str]],
    summaries: list[list[str]],
    tokenizer: PreTrainedTokenizer,
) -> float:
    """Mean summary-to-segment compression ratio across all examples and steps.

    compression_ratio = mean(summary_tokens / segment_tokens)
    Values < 1 indicate compression (desired).

    Args:
        segments: list of lists, one inner list per example
        summaries: list of lists, matching structure

    Returns:
        Mean compression ratio (float).
    """
    ratios = []
    for ex_segs, ex_sums in zip(segments, summaries):
        for seg, summ in zip(ex_segs, ex_sums):
            seg_len = len(tokenizer.encode(seg, add_special_tokens=False))
            sum_len = len(tokenizer.encode(summ, add_special_tokens=False))
            if seg_len > 0:
                ratios.append(sum_len / seg_len)
    return float(np.mean(ratios)) if ratios else float("nan")


def compute_peak_context(results: list[dict]) -> dict:
    """Compute statistics over peak context lengths.

    Args:
        results: list of inference output dicts with "peak_context_len" key

    Returns:
        {"mean": float, "max": int, "p50": float, "p95": float}
    """
    lens = np.array([r.get("peak_context_len", 0) for r in results], dtype=np.float32)
    if len(lens) == 0:
        return {"mean": 0.0, "max": 0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(lens)),
        "max": int(np.max(lens)),
        "p50": float(np.percentile(lens, 50)),
        "p95": float(np.percentile(lens, 95)),
    }


def compute_tokens_used_stats(results: list[dict]) -> dict:
    """Stats over total tokens used per example."""
    vals = np.array([r.get("total_tokens", r.get("tokens_used", 0)) for r in results],
                    dtype=np.float32)
    if len(vals) == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
    }


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    statistic=np.mean,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: sample values
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g. 0.95 → 95% CI)
        statistic: callable, applied to each resample
        seed: random seed for reproducibility

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    boots = np.array([
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lower = float(np.percentile(boots, 100 * alpha / 2))
    upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lower, upper


def accuracy_with_ci(
    correct: list[bool],
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> dict:
    """Compute accuracy and bootstrap CI.

    Returns:
        {"accuracy": float, "ci_lower": float, "ci_upper": float}
    """
    values = [float(c) for c in correct]
    acc = float(np.mean(values)) if values else 0.0
    lower, upper = bootstrap_ci(values, n_boot=n_boot, ci=ci)
    return {"accuracy": acc, "ci_lower": lower, "ci_upper": upper}
