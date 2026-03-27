"""Statistical significance tests for pairwise method comparisons."""
from __future__ import annotations
import numpy as np
from scipy import stats


def mcnemar_test(
    correct_A: list[bool],
    correct_B: list[bool],
) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether two methods differ in their per-example correctness
    on the SAME set of examples. Appropriate for paired evaluation.

    Null hypothesis: P(A correct, B wrong) = P(A wrong, B correct)

    Args:
        correct_A: boolean per-example correctness for method A
        correct_B: boolean per-example correctness for method B

    Returns:
        {
            "n": int              — number of examples
            "n_A_only": int       — examples where only A is correct
            "n_B_only": int       — examples where only B is correct
            "n_both": int         — both correct
            "n_neither": int      — both wrong
            "statistic": float    — McNemar chi-squared statistic
            "p_value": float      — two-sided p-value
            "significant": bool   — p_value < 0.05
            "advantage": str      — "A", "B", or "none"
        }
    """
    assert len(correct_A) == len(correct_B), "Lists must be same length"
    a = np.array(correct_A, dtype=bool)
    b = np.array(correct_B, dtype=bool)

    n_A_only  = int(np.sum(a & ~b))
    n_B_only  = int(np.sum(~a & b))
    n_both    = int(np.sum(a & b))
    n_neither = int(np.sum(~a & ~b))

    # Edwards' continuity-corrected McNemar statistic
    denom = n_A_only + n_B_only
    if denom == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float((abs(n_A_only - n_B_only) - 1) ** 2 / denom)
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    advantage = "none"
    if p_value < 0.05:
        advantage = "A" if n_A_only > n_B_only else "B"

    return {
        "n": len(correct_A),
        "n_A_only": n_A_only,
        "n_B_only": n_B_only,
        "n_both": n_both,
        "n_neither": n_neither,
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "advantage": advantage,
    }


def paired_bootstrap(
    scores_A: list[float],
    scores_B: list[float],
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for difference in means.

    Tests whether mean(A) - mean(B) is significantly non-zero.

    Args:
        scores_A: per-example scores for method A (e.g., 0/1 accuracy)
        scores_B: per-example scores for method B
        n_boot: number of bootstrap resamples
        seed: random seed

    Returns:
        {
            "mean_A": float
            "mean_B": float
            "mean_diff": float       — mean(A) - mean(B)
            "ci_lower": float        — 95% CI lower bound for mean_diff
            "ci_upper": float        — 95% CI upper bound for mean_diff
            "p_value": float         — fraction of bootstrap samples where diff ≤ 0
            "significant": bool      — p_value < 0.05
        }
    """
    assert len(scores_A) == len(scores_B)
    rng = np.random.default_rng(seed)
    a = np.array(scores_A, dtype=np.float64)
    b = np.array(scores_B, dtype=np.float64)
    n = len(a)

    obs_diff = float(np.mean(a) - np.mean(b))

    boot_diffs = np.array([
        np.mean(rng.choice(a, size=n, replace=True)) - np.mean(rng.choice(b, size=n, replace=True))
        for _ in range(n_boot)
    ])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    # One-sided p-value: fraction where A is NOT better than B
    p_value = float(np.mean(boot_diffs <= 0)) if obs_diff > 0 else float(np.mean(boot_diffs >= 0))

    return {
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
        "mean_diff": obs_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def compare_all_methods(
    methods: dict[str, list[bool]],
    reference: str,
) -> dict:
    """Run McNemar's test comparing all methods to a reference method.

    Args:
        methods: {"method_name": [per-example correct bool list], ...}
        reference: key in methods to compare against

    Returns:
        {"method_vs_reference": mcnemar_result_dict, ...}
    """
    ref_correct = methods[reference]
    results = {}
    for name, correct in methods.items():
        if name == reference:
            continue
        result = mcnemar_test(correct, ref_correct)
        results[f"{name}_vs_{reference}"] = result
    return results
