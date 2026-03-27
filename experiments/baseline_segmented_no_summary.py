"""Baseline B3: Segmented reasoning without summaries (full context passed)."""
from __future__ import annotations
from typing import Optional
from functools import partial

import jax
from transformers import PreTrainedTokenizer

from src.model.transformer import CausalLM
from src.inference.generation_utils import GenerationConfig, generate_text, extract_answer
from src.inference.iterative_reasoner import IterativeConfig
from src.eval.evaluator import evaluate
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset


def _segmented_no_summary(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    config: IterativeConfig,
    rng: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """Segments reasoning but passes full previous segments as context (no summaries).

    This isolates the contribution of summaries by comparing against a method
    that uses the same segment structure but no compression.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    segments: list[str] = []
    total_tokens = 0
    peak_context_len = 0

    gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        token_budget=config.token_budget,
        greedy=config.greedy,
    )

    for t in range(config.max_iterations):
        if total_tokens >= config.token_budget:
            break

        # Build prompt: question + ALL prior segments concatenated
        parts = [f"Problem: {problem.strip()}"]
        for s in segments:
            parts.append(s.strip())
        parts.append("<SEGMENT>")
        prompt = "\n".join(parts)

        ctx_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        peak_context_len = max(peak_context_len, ctx_len)

        rng, sub = jax.random.split(rng)
        gen_config.max_new_tokens = config.segment_len
        seg_text, n = generate_text(
            model, params, tokenizer, prompt, gen_config,
            tokens_used_so_far=total_tokens, rng=sub,
        )
        total_tokens += n
        segments.append(seg_text)
        if n == 0:
            break

    # Final answer
    final_parts = [f"Problem: {problem.strip()}"] + [s.strip() for s in segments]
    final_parts.append("<FINAL> The answer is:")
    final_prompt = "\n".join(final_parts)

    rng, sub = jax.random.split(rng)
    gen_config.max_new_tokens = 128
    final_text, n_f = generate_text(
        model, params, tokenizer, final_prompt, gen_config,
        tokens_used_so_far=total_tokens, rng=sub,
    )
    total_tokens += n_f

    answer = extract_answer(final_text) or extract_answer(" ".join(segments)) or ""
    return {
        "answer": answer,
        "segments": segments,
        "summaries": [],
        "total_tokens": total_tokens,
        "peak_context_len": peak_context_len,
        "final_text": final_text,
    }


def run(config: dict | None = None, out_path: str = "results/b3_segmented_no_summary.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    iter_config = IterativeConfig(
        segment_len=config.get("segment_len", 128),
        summary_len=0,
        max_iterations=config.get("max_iterations", 4),
        token_budget=config.get("token_budget", 2048),
    )

    inference_fn = partial(_segmented_no_summary, config=iter_config)

    results = evaluate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        inference_fn=inference_fn,
        n_examples=config.get("n_eval", 500),
        out_path=out_path,
        method_name="B3_segmented_no_summary",
    )
    return results


if __name__ == "__main__":
    run()
