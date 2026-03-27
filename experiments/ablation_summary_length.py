"""Ablation A2: Sweep summary_len ∈ {16, 32, 64}."""
from __future__ import annotations
import json
import os
from functools import partial

from src.eval.evaluator import evaluate
from src.inference.iterative_reasoner import run_iterative_reasoning, IterativeConfig
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset

SUMMARY_LENGTHS = [16, 32, 64]


def run(config: dict | None = None, out_path: str = "results/ablations/a2_summary_length.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    all_results = {}
    for sum_len in SUMMARY_LENGTHS:
        iter_config = IterativeConfig(
            segment_len=config.get("segment_len", 128),
            summary_len=sum_len,
            max_iterations=config.get("max_iterations", 4),
            token_budget=config.get("token_budget", 2048),
            conditioning="summary_only",
        )
        inference_fn = partial(run_iterative_reasoning, config=iter_config)
        result = evaluate(
            model=model,
            params=params,
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
            inference_fn=inference_fn,
            n_examples=config.get("n_eval", 500),
            out_path=None,
            method_name=f"A2_sum{sum_len}",
        )
        all_results[str(sum_len)] = {k: v for k, v in result.items() if k != "raw_results"}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved ablation A2 to {out_path}")
    return all_results


if __name__ == "__main__":
    run()
