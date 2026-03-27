"""Baseline B1: Vanilla CoT (uncapped, max 2048 tokens)."""
from __future__ import annotations
from functools import partial

from src.eval.evaluator import evaluate
from src.inference.vanilla_cot import generate_vanilla_cot
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset, save_result


def run(config: dict | None = None, out_path: str = "results/b1_vanilla_cot.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    inference_fn = partial(
        generate_vanilla_cot,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
    )

    results = evaluate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        inference_fn=inference_fn,
        n_examples=config.get("n_eval", 500),
        out_path=out_path,
        method_name="B1_vanilla_cot",
    )
    return results


if __name__ == "__main__":
    run()
