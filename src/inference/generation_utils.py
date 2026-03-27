"""Shared generation utilities and token budget tracking."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from transformers import PreTrainedTokenizer

from src.model.transformer import CausalLM


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    token_budget: int = 2048      # hard cap on total tokens generated across all steps
    greedy: bool = False          # if True, ignores temperature/top_p


def greedy_decode(
    model: CausalLM,
    params,
    input_ids: np.ndarray,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> tuple[str, int]:
    """Greedy autoregressive decoding.

    Args:
        model: CausalLM instance
        params: model parameters pytree
        input_ids: (1, prompt_len) int32 array
        max_new_tokens: max tokens to generate
        eos_token_id: stop on this token if set

    Returns:
        (generated_text_only, n_tokens_generated)
        Note: returns only the newly generated tokens decoded, not the prompt.
    """
    ids = jnp.array(input_ids)
    prompt_len = ids.shape[1]

    generated_ids, n_gen = model.generate(
        params,
        ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        eos_token_id=eos_token_id,
    )

    # Return only the newly generated portion
    new_ids = np.array(generated_ids[0, prompt_len:])
    return new_ids, n_gen


def nucleus_sample(
    model: CausalLM,
    params,
    input_ids: np.ndarray,
    config: GenerationConfig,
    eos_token_id: Optional[int] = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> tuple[np.ndarray, int]:
    """Nucleus (top-p) sampling.

    Returns:
        (new_token_ids, n_tokens_generated)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    ids = jnp.array(input_ids)
    prompt_len = ids.shape[1]

    generated_ids, n_gen = model.generate(
        params,
        ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        eos_token_id=eos_token_id,
        rng=rng,
    )

    new_ids = np.array(generated_ids[0, prompt_len:])
    return new_ids, n_gen


def generate_text(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig,
    tokens_used_so_far: int = 0,
    rng: Optional[jax.random.PRNGKey] = None,
) -> tuple[str, int]:
    """Generate text from a prompt, respecting the token budget.

    Args:
        model: CausalLM instance
        params: model parameters
        tokenizer: HuggingFace tokenizer
        prompt: input text
        config: GenerationConfig
        tokens_used_so_far: tokens already consumed (for budget tracking)
        rng: JAX PRNG key

    Returns:
        (generated_text, n_new_tokens)
    """
    remaining_budget = config.token_budget - tokens_used_so_far
    max_new = min(config.max_new_tokens, max(0, remaining_budget))
    if max_new == 0:
        return "", 0

    input_ids = np.array(
        tokenizer.encode(prompt, add_special_tokens=False), dtype=np.int32
    )[None, :]  # (1, T)

    if config.greedy or config.temperature == 0.0:
        new_ids, n_gen = greedy_decode(
            model, params, input_ids, max_new, tokenizer.eos_token_id
        )
    else:
        new_ids, n_gen = nucleus_sample(
            model, params, input_ids, config, tokenizer.eos_token_id, rng
        )

    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, n_gen


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Common patterns for math answer extraction
_ANSWER_PATTERNS = [
    r"\\boxed\{([^}]+)\}",                    # LaTeX boxed
    r"[Tt]he answer is[:\s]+([^\n.]+)",        # "The answer is X"
    r"[Aa]nswer[:\s=]+([^\n.]+)",              # "Answer: X"
    r"=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$", # trailing = <number>
    r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*$",     # trailing number
]


def extract_answer(text: str) -> Optional[str]:
    """Extract a final answer from generated text.

    Tries patterns in order of specificity. Returns the first match,
    stripped of whitespace, or None if nothing matches.
    """
    for pattern in _ANSWER_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def token_count(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """Count tokens in a string."""
    return len(tokenizer.encode(text, add_special_tokens=False))
