"""Vanilla chain-of-thought inference (Baseline B1 and B2)."""
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


def generate_vanilla_cot(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    greedy: bool = False,
    rng: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """Generate a single long chain-of-thought reasoning trace.

    Baseline B1: max_tokens=2048 (uncapped)
    Baseline B2: max_tokens=matched budget of InftyThink run

    Args:
        model: CausalLM instance
        params: model parameters
        tokenizer: HuggingFace tokenizer
        problem: problem statement
        max_tokens: total token cap for the reasoning + answer
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        greedy: if True, use greedy decoding
        rng: JAX PRNG key

    Returns:
        {
            "answer":      str   — extracted final answer (or "" if none found)
            "reasoning":   str   — full generated reasoning trace
            "tokens_used": int   — tokens generated (reasoning + answer)
            "prompt_tokens": int — tokens in the prompt
        }
    """
    prompt = _build_cot_prompt(problem)
    prompt_tokens = token_count(prompt, tokenizer)

    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        token_budget=max_tokens,
        greedy=greedy,
    )

    reasoning, n_gen = generate_text(
        model, params, tokenizer, prompt, config, rng=rng
    )

    answer = extract_answer(reasoning) or ""

    return {
        "answer": answer,
        "reasoning": reasoning,
        "tokens_used": n_gen,
        "total_tokens": n_gen,
        "prompt_tokens": prompt_tokens,
        # For vanilla CoT the full prompt+generation is the context at each token
        "peak_context_len": prompt_tokens + n_gen,
    }


def _build_cot_prompt(problem: str) -> str:
    """Build the standard CoT prompt."""
    return (
        f"Problem: {problem.strip()}\n\n"
        "Let me solve this step by step.\n"
    )
