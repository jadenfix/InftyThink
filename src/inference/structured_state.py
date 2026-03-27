"""Extension E1: Structured progress state reasoning (Phase C)."""
from __future__ import annotations
from typing import Optional

import jax
from transformers import PreTrainedTokenizer

from src.model.transformer import CausalLM
from src.inference.generation_utils import (
    GenerationConfig,
    generate_text,
    extract_answer,
    token_count,
)
from src.inference.iterative_reasoner import IterativeConfig
from src.data.summary_generator import (
    init_structured_state,
    structured_summary,
    state_to_text,
    verify_state_consistency,
)


def run_structured_iterative(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    config: IterativeConfig,
    rng: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """Structured-state iterative reasoning loop (Extension E1).

    Identical protocol to run_iterative_reasoning, but replaces free-form
    text summaries with a structured state dict:
        {known_facts, open_subgoals, derived_values, constraints, confidence}

    The state is serialized to text for injection into the context, and a
    lightweight verifier checks constraint / value preservation at each step.

    Returns same dict as run_iterative_reasoning plus:
        "states":          list[dict]  — structured state at each iteration
        "consistency_checks": list[dict]  — verifier result at each step
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    segments: list[str] = []
    states: list[dict] = []
    consistency_checks: list[dict] = []
    tokens_per_step: list[int] = []
    peak_context_len: int = 0
    total_tokens: int = 0

    current_state = init_structured_state()

    gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        token_budget=config.token_budget,
        greedy=config.greedy,
    )

    for t in range(config.max_iterations):
        if total_tokens >= config.token_budget:
            break

        # --- Build segment prompt with structured state ---
        segment_prompt = _build_structured_segment_prompt(
            problem, states, config
        )
        ctx_len = token_count(segment_prompt, tokenizer)
        peak_context_len = max(peak_context_len, ctx_len)

        rng, sub = jax.random.split(rng)
        gen_config.max_new_tokens = config.segment_len
        segment_text, n_seg = generate_text(
            model, params, tokenizer, segment_prompt, gen_config,
            tokens_used_so_far=total_tokens, rng=sub,
        )
        total_tokens += n_seg
        tokens_per_step.append(n_seg)
        segments.append(segment_text)

        if total_tokens >= config.token_budget or n_seg == 0:
            break

        # --- Update structured state from segment ---
        prior_state = current_state
        current_state = structured_summary(
            segment=segment_text,
            prior_state=prior_state,
            problem=problem if t == 0 else "",
        )

        # Verifier: check consistency
        check = verify_state_consistency(prior_state, current_state)
        consistency_checks.append(check)
        states.append(dict(current_state))

        # Count tokens for state serialization
        state_text = state_to_text(current_state)
        total_tokens += token_count(state_text, tokenizer)
        tokens_per_step.append(token_count(state_text, tokenizer))

    # --- Final answer step ---
    final_prompt = _build_structured_final_prompt(problem, states)
    ctx_len = token_count(final_prompt, tokenizer)
    peak_context_len = max(peak_context_len, ctx_len)

    rng, sub = jax.random.split(rng)
    gen_config.max_new_tokens = 128
    final_text, n_final = generate_text(
        model, params, tokenizer, final_prompt, gen_config,
        tokens_used_so_far=total_tokens, rng=sub,
    )
    total_tokens += n_final
    tokens_per_step.append(n_final)

    answer = extract_answer(final_text) or extract_answer(" ".join(segments)) or ""

    return {
        "answer": answer,
        "segments": segments,
        "summaries": [state_to_text(s) for s in states],
        "states": states,
        "consistency_checks": consistency_checks,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "n_iterations": len(segments),
        "peak_context_len": peak_context_len,
        "final_text": final_text,
    }


def _build_structured_segment_prompt(
    problem: str,
    states: list[dict],
    config: IterativeConfig,
) -> str:
    parts = [f"Problem: {problem.strip()}"]
    if states:
        latest = states[-1]
        parts.append("[Progress State]")
        parts.append(state_to_text(latest))
    parts.append("<SEGMENT>")
    return "\n".join(parts)


def _build_structured_final_prompt(
    problem: str,
    states: list[dict],
) -> str:
    parts = [f"Problem: {problem.strip()}"]
    if states:
        parts.append("[Final Progress State]")
        parts.append(state_to_text(states[-1]))
    parts.append("<FINAL> The answer is:")
    return "\n".join(parts)
