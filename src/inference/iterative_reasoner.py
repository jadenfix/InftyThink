"""InftyThink iterative reasoning loop (Main Method M1 and ablations)."""
from __future__ import annotations
from dataclasses import dataclass
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


@dataclass
class IterativeConfig:
    segment_len: int = 128       # K: max tokens per reasoning segment
    summary_len: int = 32        # S: max tokens per summary
    max_iterations: int = 4      # T: max reasoning iterations
    token_budget: int = 2048     # hard cap on total generated tokens
    conditioning: str = "summary_only"
    # conditioning options:
    #   "summary_only"     — context = [question] + all prior summaries
    #   "summary+tail"     — context = [question] + all prior summaries + tail of last segment
    #   "all_summaries"    — same as summary_only (alias)
    #   "rolling_state"    — context = [question] + only the latest summary
    tail_len: int = 32           # tokens from latest segment tail (for summary+tail mode)
    temperature: float = 0.7
    top_p: float = 0.95
    greedy: bool = False


def run_iterative_reasoning(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    config: IterativeConfig,
    rng: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """InftyThink segment-summary reasoning loop.

    Protocol:
      For t = 1, ..., T:
        1. Build context from [question] + prior summaries (+ optional tail)
        2. Generate a reasoning segment of ≤ K tokens
        3. Generate a summary of ≤ S tokens
        4. If token budget exhausted, stop early
      5. Generate final answer from [question] + all summaries

    Args:
        model: CausalLM instance
        params: model parameters
        tokenizer: HuggingFace tokenizer
        problem: problem statement string
        config: IterativeConfig
        rng: JAX PRNG key

    Returns:
        {
            "answer":           str
            "segments":         list[str]   — one per iteration
            "summaries":        list[str]   — one per iteration
            "tokens_per_step":  list[int]   — tokens generated at each step
            "total_tokens":     int
            "n_iterations":     int         — actual iterations completed
            "peak_context_len": int         — max prompt token count seen
        }
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    segments: list[str] = []
    summaries: list[str] = []
    tokens_per_step: list[int] = []
    peak_context_len: int = 0
    total_tokens: int = 0

    gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        token_budget=config.token_budget,
        greedy=config.greedy,
    )

    for t in range(config.max_iterations):
        if total_tokens >= config.token_budget:
            break

        # --- Build segment prompt ---
        segment_prompt = _build_segment_prompt(
            problem, summaries, segments, config
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

        # --- Build summary prompt ---
        summary_prompt = _build_summary_prompt(
            problem, summaries, segment_text, config
        )
        ctx_len = token_count(summary_prompt, tokenizer)
        peak_context_len = max(peak_context_len, ctx_len)

        rng, sub = jax.random.split(rng)
        gen_config.max_new_tokens = config.summary_len
        summary_text, n_sum = generate_text(
            model, params, tokenizer, summary_prompt, gen_config,
            tokens_used_so_far=total_tokens, rng=sub,
        )
        total_tokens += n_sum
        tokens_per_step.append(n_sum)
        summaries.append(summary_text)

    # --- Final answer step ---
    final_prompt = _build_final_prompt(problem, summaries)
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
        "summaries": summaries,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "n_iterations": len(segments),
        "peak_context_len": peak_context_len,
        "final_text": final_text,
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_segment_prompt(
    problem: str,
    summaries: list[str],
    segments: list[str],
    config: IterativeConfig,
) -> str:
    parts = [f"Problem: {problem.strip()}"]

    if config.conditioning in ("summary_only", "all_summaries"):
        for i, s in enumerate(summaries):
            parts.append(f"<SUMMARY> {s.strip()}")

    elif config.conditioning == "summary+tail":
        for s in summaries[:-1]:
            parts.append(f"<SUMMARY> {s.strip()}")
        if summaries:
            parts.append(f"<SUMMARY> {summaries[-1].strip()}")
        if segments:
            # Append tail of the last segment (last tail_len chars as approximation)
            last_seg = segments[-1]
            parts.append(f"[...] {last_seg[-config.tail_len * 4:]}")

    elif config.conditioning == "rolling_state":
        # Only the most recent summary
        if summaries:
            parts.append(f"<SUMMARY> {summaries[-1].strip()}")

    parts.append("<SEGMENT>")
    return "\n".join(parts)


def _build_summary_prompt(
    problem: str,
    summaries: list[str],
    latest_segment: str,
    config: IterativeConfig,
) -> str:
    parts = [f"Problem: {problem.strip()}"]
    for s in summaries:
        parts.append(f"<SUMMARY> {s.strip()}")
    parts.append(latest_segment.strip())
    parts.append("<SUMMARY>")
    return "\n".join(parts)


def _build_final_prompt(problem: str, summaries: list[str]) -> str:
    parts = [f"Problem: {problem.strip()}"]
    for s in summaries:
        parts.append(f"<SUMMARY> {s.strip()}")
    parts.append("<FINAL> The answer is:")
    return "\n".join(parts)
