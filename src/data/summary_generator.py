"""Heuristic and structured summary generation from reasoning segments."""
from __future__ import annotations
import re
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Heuristic (extractive) summaries
# ---------------------------------------------------------------------------

def heuristic_summary(
    segment: str,
    tokenizer: PreTrainedTokenizer,
    max_summary_tokens: int = 32,
) -> str:
    """Extractive summary: first + last sentence of the segment, truncated.

    Strategy:
      1. Split the segment into sentences.
      2. Keep the first sentence (problem framing) + last sentence (latest result).
      3. Truncate the concatenation to max_summary_tokens.

    Args:
        segment: Raw text of one reasoning segment.
        tokenizer: Used for token-count truncation.
        max_summary_tokens: Hard token cap on the returned summary.

    Returns:
        A short summary string.
    """
    sentences = _split_sentences(segment)
    if not sentences:
        return _truncate(segment, tokenizer, max_summary_tokens)
    if len(sentences) == 1:
        return _truncate(sentences[0], tokenizer, max_summary_tokens)

    candidate = sentences[0].strip() + " " + sentences[-1].strip()
    return _truncate(candidate, tokenizer, max_summary_tokens)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on ., !, ? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _truncate(text: str, tokenizer: PreTrainedTokenizer, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens, return decoded string."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Structured state summaries
# ---------------------------------------------------------------------------

# Schema for the structured progress state (used by Phase C extension).
STATE_SCHEMA: dict = {
    "known_facts": list,       # confirmed facts derived so far
    "open_subgoals": list,     # remaining steps yet to solve
    "derived_values": dict,    # symbol/name → numeric or symbolic value
    "constraints": list,       # problem constraints that must be preserved
    "confidence": float,       # 0.0–1.0 self-assessed confidence
}


def init_structured_state() -> dict:
    """Return an empty structured state."""
    return {
        "known_facts": [],
        "open_subgoals": [],
        "derived_values": {},
        "constraints": [],
        "confidence": 0.5,
    }


def structured_summary(
    segment: str,
    prior_state: dict,
    problem: str = "",
) -> dict:
    """Heuristic update of a structured state from a new segment.

    This is a rule-based approximation used before model training.
    It extracts:
      - numeric assignments (x = <number>) → derived_values
      - "therefore" / "so" sentences → known_facts
      - "need to" / "must" / "next" sentences → open_subgoals
      - sentences containing "where" / "given" / "constraint" → constraints

    Args:
        segment: The latest reasoning segment text.
        prior_state: The state dict from the previous step.
        problem: The original problem text (used to seed constraints on step 0).

    Returns:
        Updated state dict (does NOT modify prior_state in-place).
    """
    state = {
        "known_facts": list(prior_state.get("known_facts", [])),
        "open_subgoals": list(prior_state.get("open_subgoals", [])),
        "derived_values": dict(prior_state.get("derived_values", {})),
        "constraints": list(prior_state.get("constraints", [])),
        "confidence": prior_state.get("confidence", 0.5),
    }

    sentences = _split_sentences(segment)

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue

        # Extract numeric assignments: "x = 3.14" or "n = 42"
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([\d.]+(?:/[\d.]+)?)", s):
            state["derived_values"][m.group(1)] = m.group(2)

        # Known facts: sentences with conclusion markers
        lower = s.lower()
        if any(marker in lower for marker in ("therefore", "thus", "so ", "hence", "we get", "we have")):
            if s not in state["known_facts"]:
                state["known_facts"].append(s)

        # Open subgoals: sentences with forward-looking markers
        if any(marker in lower for marker in ("need to", "must", "next,", "now we", "it remains", "still need")):
            if s not in state["open_subgoals"]:
                state["open_subgoals"].append(s)

        # Constraints: given conditions
        if any(marker in lower for marker in ("given", "where ", "constraint", "assume", "let ")):
            if s not in state["constraints"]:
                state["constraints"].append(s)

    # Seed constraints from problem text on first pass
    if problem and not state["constraints"]:
        for sent in _split_sentences(problem):
            s = sent.strip()
            if any(m in s.lower() for m in ("given", "where ", "if ", "suppose", "assume")):
                state["constraints"].append(s)

    # Heuristic confidence: increases with more known_facts and fewer open_subgoals
    n_facts = len(state["known_facts"])
    n_goals = max(len(state["open_subgoals"]), 1)
    state["confidence"] = min(1.0, 0.3 + 0.1 * n_facts / n_goals)

    return state


def state_to_text(state: dict, max_items: int = 3) -> str:
    """Serialize a structured state to a compact text representation.

    Used to inject the state into the model's context window.
    Limits each field to max_items entries to control token usage.
    """
    lines = []
    if state.get("known_facts"):
        facts = state["known_facts"][-max_items:]
        lines.append("known_facts: " + "; ".join(facts))
    if state.get("open_subgoals"):
        goals = state["open_subgoals"][-max_items:]
        lines.append("open_subgoals: " + "; ".join(goals))
    if state.get("derived_values"):
        items = list(state["derived_values"].items())[-max_items:]
        lines.append("derived_values: " + ", ".join(f"{k}={v}" for k, v in items))
    if state.get("constraints"):
        cons = state["constraints"][-max_items:]
        lines.append("constraints: " + "; ".join(cons))
    lines.append(f"confidence: {state.get('confidence', 0.5):.2f}")
    return "\n".join(lines)


def verify_state_consistency(prior_state: dict, new_state: dict) -> dict:
    """Check that new_state preserves key invariants from prior_state.

    Returns:
        {
            "constraints_preserved": bool,
            "values_preserved": bool,
            "missing_constraints": list[str],
            "drifted_values": dict,
        }
    """
    missing_constraints = []
    for c in prior_state.get("constraints", []):
        # A constraint is "preserved" if it appears verbatim or as a substring
        found = any(c in nc or nc in c for nc in new_state.get("constraints", []))
        if not found:
            missing_constraints.append(c)

    drifted_values = {}
    for k, v in prior_state.get("derived_values", {}).items():
        if k in new_state.get("derived_values", {}):
            new_v = new_state["derived_values"][k]
            if new_v != v:
                drifted_values[k] = {"before": v, "after": new_v}

    return {
        "constraints_preserved": len(missing_constraints) == 0,
        "values_preserved": len(drifted_values) == 0,
        "missing_constraints": missing_constraints,
        "drifted_values": drifted_values,
    }
