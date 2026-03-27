"""Answer normalization and matching for math problems."""
from __future__ import annotations
import re
from fractions import Fraction


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for comparison.

    Steps:
      1. Strip whitespace
      2. Lowercase
      3. Remove LaTeX wrappers (\\frac, \\sqrt, $...$)
      4. Normalize fractions (3/4 → 0.75)
      5. Normalize decimals (0.750 → 0.75)
      6. Remove trailing zeros after decimal point
    """
    if ans is None:
        return ""
    s = ans.strip().lower()
    # Strip LaTeX dollar signs
    s = re.sub(r"\$+", "", s)
    # Strip \\boxed{...}
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    # Strip \\text{...}
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    # Strip commas in numbers (1,000 → 1000)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = s.strip()
    # Try to normalize fractions to decimals for comparison
    frac_match = re.fullmatch(r"(-?)(\d+)/(\d+)", s)
    if frac_match:
        sign, num, den = frac_match.groups()
        if int(den) != 0:
            val = int(num) / int(den)
            s = f"{'-' if sign else ''}{val:.10f}".rstrip("0").rstrip(".")
    # Try to parse as float and normalize
    else:
        try:
            val = float(s)
            # Re-format to remove trailing zeros
            s = f"{val:.10f}".rstrip("0").rstrip(".")
        except ValueError:
            pass
    return s.strip()


def answers_match(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Check whether predicted and gold answers match.

    Matching strategy (in order):
      1. Exact string match after normalization
      2. Numeric match within tolerance tol
      3. Fraction equivalence (e.g. 1/2 == 0.5)

    Args:
        pred: predicted answer string
        gold: gold answer string
        tol: absolute tolerance for numeric comparison

    Returns:
        True if answers are considered equivalent.
    """
    if pred is None or gold is None:
        return False

    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)

    if norm_pred == norm_gold:
        return True

    # Try numeric comparison
    try:
        p_val = float(norm_pred)
        g_val = float(norm_gold)
        if abs(p_val - g_val) <= tol:
            return True
        # Relative tolerance for large numbers
        if g_val != 0 and abs((p_val - g_val) / g_val) <= tol:
            return True
    except ValueError:
        pass

    # Try fraction parsing
    try:
        p_frac = Fraction(pred.strip()).limit_denominator(10_000)
        g_frac = Fraction(gold.strip()).limit_denominator(10_000)
        if abs(float(p_frac) - float(g_frac)) <= tol:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    return False


def batch_evaluate_answers(
    predictions: list[str | None],
    gold_answers: list[str],
    tol: float = 1e-6,
) -> list[bool]:
    """Evaluate a list of predictions against gold answers.

    Returns:
        List of bool, True = correct match.
    """
    return [
        answers_match(pred, gold, tol)
        for pred, gold in zip(predictions, gold_answers)
    ]
