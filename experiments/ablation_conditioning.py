"""Ablation A4: Conditioning strategy sweep.

Conditions:
  summary_only    — context = [question] + all prior summaries
  summary+tail    — context = [question] + all prior summaries + tail of last segment
  all_summaries   — alias for summary_only
  rolling_state   — context = [question] + only the latest summary
"""
from __future__ import annotations
import json
import os
from functools import partial

from src.eval.evaluator import evaluate
from src.inference.iterative_reasoner import run_iterative_reasoning, IterativeConfig
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset

CONDITIONING_STRATEGIES = ["summary_only", "summary+tail", "rolling_state"]


def run(config: dict | None = None, out_path: str = "results/ablations/a4_conditioning.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    all_results = {}
    for cond in CONDITIONING_STRATEGIES:
        iter_config = IterativeConfig(
            segment_len=config.get("segment_len", 128),
            summary_len=config.get("summary_len", 32),
            max_iterations=config.get("max_iterations", 4),
            token_budget=config.get("token_budget", 2048),
            conditioning=cond,
            tail_len=config.get("tail_len", 32),
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
            method_name=f"A4_{cond}",
        )
        all_results[cond] = {k: v for k, v in result.items() if k != "raw_results"}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved ablation A4 to {out_path}")
    return all_results


if __name__ == "__main__":
    run()
