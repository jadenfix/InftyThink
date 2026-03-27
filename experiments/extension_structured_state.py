"""Extension E1: Structured progress state reasoning."""
from __future__ import annotations
import json
import os
from functools import partial

from src.eval.evaluator import evaluate
from src.inference.structured_state import run_structured_iterative
from src.inference.iterative_reasoner import IterativeConfig
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset


def run(config: dict | None = None, out_path: str = "results/e1_structured_state.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    iter_config = IterativeConfig(
        segment_len=config.get("segment_len", 128),
        summary_len=config.get("summary_len", 32),
        max_iterations=config.get("max_iterations", 4),
        token_budget=config.get("token_budget", 2048),
        conditioning="summary_only",
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.95),
    )

    inference_fn = partial(run_structured_iterative, config=iter_config)

    results = evaluate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        inference_fn=inference_fn,
        n_examples=config.get("n_eval", 500),
        out_path=out_path,
        method_name="E1_structured_state",
    )
    return results


if __name__ == "__main__":
    run()
