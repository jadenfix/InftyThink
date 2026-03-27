"""Baseline B2: Token-capped CoT (matched budget = InftyThink default)."""
from __future__ import annotations
from functools import partial

from src.eval.evaluator import evaluate
from src.inference.vanilla_cot import generate_vanilla_cot
from src.model.tokenizer import load_tokenizer
from experiments._base import load_model_and_params, load_eval_dataset

# Match InftyThink default budget:
# max_iterations=4, segment_len=128, summary_len=32 → 4*(128+32)+128 = 768
# plus overhead, cap at 2048 same as B1 but track separately
CAPPED_BUDGET = 768  # 4 * (128 + 32) = 640 + 128 final = 768 tokens


def run(config: dict | None = None, out_path: str = "results/b2_capped_cot.json") -> dict:
    config = config or {}
    model, params, model_config = load_model_and_params()
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=config.get("n_eval", 500))

    budget = config.get("token_budget", CAPPED_BUDGET)
    inference_fn = partial(
        generate_vanilla_cot,
        max_tokens=budget,
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
        method_name="B2_capped_cot",
    )
    return results


if __name__ == "__main__":
    run()
