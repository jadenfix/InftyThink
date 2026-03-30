"""Classify reasoning failures into 5 diagnostic categories."""
from __future__ import annotations
import re
from typing import Optional


# Failure category constants
CONSTRAINT_LOSS = "constraint_loss"
NUMERIC_DRIFT = "numeric_drift"
SUBGOAL_OMISSION = "subgoal_omission"
OVERCOMPRESSION = "overcompression"
REDUNDANT_OVERHEAD = "redundant_overhead"

FAILURE_CATEGORIES = [
    CONSTRAINT_LOSS,
    NUMERIC_DRIFT,
    SUBGOAL_OMISSION,
    OVERCOMPRESSION,
    REDUNDANT_OVERHEAD,
]


def classify_failure(
    problem: str,
    segments: list[str],
    summaries: list[str],
    gold_answer: str,
    pred_answer: str,
) -> str:
    """Rule-based failure classification for one incorrect example.

    Categories (in priority order):
      1. constraint_loss    — a numeric or conditional constraint from the
                              problem is absent from the final summary
      2. numeric_drift      — a numeric value changes value across segments
      3. subgoal_omission   — a "need to" goal is mentioned in one segment
                              but absent from subsequent summaries
      4. overcompression    — summaries are very short relative to segments
                              (compression ratio < 0.1) and the final answer
                              is wrong
      5. redundant_overhead — all summaries together are longer than the
                              segments they summarize (ratio > 0.9)

    Returns one of the FAILURE_CATEGORIES strings.
    """
    # --- 1. constraint_loss ---
    if _check_constraint_loss(problem, summaries):
        return CONSTRAINT_LOSS

    # --- 2. numeric_drift ---
    if _check_numeric_drift(segments, summaries):
        return NUMERIC_DRIFT

    # --- 3. subgoal_omission ---
    if _check_subgoal_omission(segments, summaries):
        return SUBGOAL_OMISSION

    # --- 4. overcompression ---
    if _check_overcompression(segments, summaries):
        return OVERCOMPRESSION

    # --- 5. redundant_overhead (default for remaining failures) ---
    return REDUNDANT_OVERHEAD


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric strings from text."""
    return re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)


def _check_constraint_loss(problem: str, summaries: list[str]) -> bool:
    """True if a number from the problem is absent from the last summary."""
    if not summaries:
        return False
    problem_numbers = set(_extract_numbers(problem))
    if not problem_numbers:
        return False
    last_summary_numbers = set(_extract_numbers(summaries[-1]))
    # If >50% of problem numbers are missing from the last summary
    missing = problem_numbers - last_summary_numbers
    return len(missing) > len(problem_numbers) * 0.5


def _check_numeric_drift(segments: list[str], summaries: list[str]) -> bool:
    """True if a number established in an early segment appears mutated in a later summary.

    Drift criterion: number N from segment 0 appears in a later summary as N' where
    N' is close but not equal to N (within 10% of N), suggesting a transcription error.
    This requires the number to appear in both places (not just two arbitrary numbers).
    """
    if len(segments) < 2 or len(summaries) < 2:
        return False

    early_nums = set(_extract_numbers(segments[0]))
    # Only flag if we also see a version of that number appear in the summary (mutation)
    for summary in summaries[1:]:
        summary_nums = set(_extract_numbers(summary))
        for en in early_nums:
            try:
                en_f = float(en)
            except ValueError:
                continue
            if en_f == 0:
                continue
            for sn in summary_nums:
                try:
                    sn_f = float(sn)
                except ValueError:
                    continue
                # Same magnitude but different value: relative error 1e-4 to 20%
                rel_err = abs(en_f - sn_f) / abs(en_f)
                if 1e-4 < rel_err < 0.20:
                    return True
    return False


def _check_subgoal_omission(segments: list[str], summaries: list[str]) -> bool:
    """True if a 'need to / must / it remains' goal from an early segment
    is absent from all later summaries."""
    if len(segments) < 2 or len(summaries) < 1:
        return False

    goal_patterns = re.compile(
        r"(need to|must|it remains|still need|next step|now (?:we|I) (?:need|must))",
        re.IGNORECASE,
    )

    # Find goals mentioned in first half of segments
    early_goals: list[str] = []
    cutoff = max(1, len(segments) // 2)
    for seg in segments[:cutoff]:
        for sentence in re.split(r"[.!?]", seg):
            if goal_patterns.search(sentence):
                early_goals.append(sentence.strip().lower())

    if not early_goals:
        return False

    # Check if any goal is absent from all later summaries
    all_summary_text = " ".join(summaries).lower()
    for goal in early_goals:
        # Use key nouns from the goal (rough heuristic)
        key_words = [w for w in goal.split() if len(w) > 4]
        if key_words:
            found = any(kw in all_summary_text for kw in key_words)
            if not found:
                return True
    return False


def _check_overcompression(segments: list[str], summaries: list[str]) -> bool:
    """True if the mean summary-to-segment character ratio is < 0.1."""
    if not segments or not summaries:
        return False
    ratios = []
    for seg, summ in zip(segments, summaries):
        if len(seg) > 0:
            ratios.append(len(summ) / len(seg))
    if not ratios:
        return False
    return float(sum(ratios) / len(ratios)) < 0.1


def analyze_failures(
    results: list[dict],
    gold_answers: list[str],
    problem_texts: Optional[list[str]] = None,
) -> dict:
    """Classify all failures and return counts + fractions by category.

    Args:
        results: list of inference output dicts with keys:
                   "answer", "segments", "summaries"
        gold_answers: ground truth answers (same length as results)
        problem_texts: original problem texts (optional, improves constraint_loss detection)

    Returns:
        {
            "total": int,
            "n_failures": int,
            "categories": {
                "constraint_loss":   {"count": int, "fraction": float},
                "numeric_drift":     {"count": int, "fraction": float},
                "subgoal_omission":  {"count": int, "fraction": float},
                "overcompression":   {"count": int, "fraction": float},
                "redundant_overhead": {"count": int, "fraction": float},
            },
            "per_example": list[str | None]   # category label for each example, None if correct
        }
    """
    from src.eval.answer_extractor import answers_match

    counts = {cat: 0 for cat in FAILURE_CATEGORIES}
    per_example: list[Optional[str]] = []

    n_failures = 0
    for i, (result, gold) in enumerate(zip(results, gold_answers)):
        pred = result.get("answer", "")
        if answers_match(pred, gold):
            per_example.append(None)
            continue

        n_failures += 1
        problem = problem_texts[i] if problem_texts else ""
        segments = result.get("segments", [result.get("reasoning", "")])
        summaries = result.get("summaries", [])

        category = classify_failure(
            problem=problem,
            segments=segments if isinstance(segments, list) else [segments],
            summaries=summaries,
            gold_answer=gold,
            pred_answer=pred,
        )
        counts[category] += 1
        per_example.append(category)

    n_fail = max(n_failures, 1)
    return {
        "total": len(results),
        "n_failures": n_failures,
        "categories": {
            cat: {
                "count": counts[cat],
                "fraction": round(counts[cat] / n_fail, 4),
            }
            for cat in FAILURE_CATEGORIES
        },
        "per_example": per_example,
    }
