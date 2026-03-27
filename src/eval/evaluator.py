"""Run full evaluation of an inference method on the eval dataset."""
from __future__ import annotations
import json
import os
import time
from typing import Callable, Optional

import jax
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.model.transformer import CausalLM
from src.eval.answer_extractor import answers_match
from src.eval.metrics import (
    compute_accuracy,
    compute_token_efficiency,
    compute_compression_ratio,
    compute_peak_context,
    compute_tokens_used_stats,
    accuracy_with_ci,
    bootstrap_ci,
)
from src.eval.failure_analyzer import analyze_failures


def evaluate(
    model: CausalLM,
    params,
    tokenizer: PreTrainedTokenizer,
    eval_dataset,
    inference_fn: Callable,
    n_examples: int = 500,
    out_path: Optional[str] = None,
    method_name: str = "unknown",
    seed: int = 42,
) -> dict:
    """Run model on eval_dataset and compute all metrics.

    Args:
        model: CausalLM instance
        params: model parameters
        tokenizer: HuggingFace tokenizer
        eval_dataset: HuggingFace Dataset with columns: problem, answer
        inference_fn: callable(model, params, tokenizer, problem) → dict
                      The returned dict must have keys: "answer", optionally
                      "segments", "summaries", "total_tokens", "peak_context_len"
        n_examples: number of examples to evaluate (≤ len(eval_dataset))
        out_path: if set, save results JSON here
        method_name: label for logging
        seed: random seed (shuffles eval subset for reproducibility)

    Returns:
        Full metrics dict:
        {
            "method":              str
            "n_examples":          int
            "accuracy":            float
            "accuracy_ci_lower":   float
            "accuracy_ci_upper":   float
            "token_efficiency":    float   (mean tokens per correct answer)
            "compression_ratio":   float   (mean summary/segment length)
            "peak_context_mean":   float
            "peak_context_p95":    float
            "tokens_used_mean":    float
            "tokens_used_p95":     float
            "failure_breakdown":   dict
            "wall_clock_seconds":  float
            "raw_results":         list[dict]
        }
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eval_dataset), size=min(n_examples, len(eval_dataset)), replace=False)
    subset = eval_dataset.select(indices.tolist())

    results: list[dict] = []
    predictions: list[Optional[str]] = []
    gold_answers: list[str] = [row["answer"] for row in subset]
    problem_texts: list[str] = [row["problem"] for row in subset]

    print(f"\nEvaluating [{method_name}] on {len(subset)} examples...")
    t0 = time.time()

    jax_rng = jax.random.PRNGKey(seed)
    for i, row in enumerate(tqdm(subset, desc=method_name)):
        jax_rng, sub_rng = jax.random.split(jax_rng)
        try:
            result = inference_fn(
                model, params, tokenizer, row["problem"], rng=sub_rng
            )
        except Exception as e:
            result = {
                "answer": "",
                "segments": [],
                "summaries": [],
                "total_tokens": 0,
                "peak_context_len": 0,
                "error": str(e),
            }
        result["gold_answer"] = row["answer"]
        result["problem"] = row["problem"]
        results.append(result)
        predictions.append(result.get("answer", ""))

    wall_clock = time.time() - t0

    # Correctness flags
    correct = [answers_match(p, g) for p, g in zip(predictions, gold_answers)]

    # Core metrics
    acc_info = accuracy_with_ci(correct, n_boot=10_000, ci=0.95)
    tokens_used = [r.get("total_tokens", r.get("tokens_used", 0)) for r in results]
    eff = compute_token_efficiency(tokens_used, correct)
    segs = [r.get("segments", [r.get("reasoning", "")]) for r in results]
    sums = [r.get("summaries", []) for r in results]
    segs_lists = [[s] if isinstance(s, str) else s for s in segs]
    comp = compute_compression_ratio(segs_lists, sums, tokenizer) if any(sums) else float("nan")
    peak = compute_peak_context(results)
    tok_stats = compute_tokens_used_stats(results)
    failures = analyze_failures(results, gold_answers, problem_texts)

    metrics = {
        "method": method_name,
        "n_examples": len(subset),
        "accuracy": acc_info["accuracy"],
        "accuracy_ci_lower": acc_info["ci_lower"],
        "accuracy_ci_upper": acc_info["ci_upper"],
        "token_efficiency": eff,
        "compression_ratio": comp,
        "peak_context_mean": peak["mean"],
        "peak_context_p95": peak["p95"],
        "tokens_used_mean": tok_stats["mean"],
        "tokens_used_p95": tok_stats["p95"],
        "failure_breakdown": failures["categories"],
        "n_failures": failures["n_failures"],
        "wall_clock_seconds": round(wall_clock, 2),
        "raw_results": results,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # Don't serialize raw_results to the summary JSON (too large)
        summary = {k: v for k, v in metrics.items() if k != "raw_results"}
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        # Save raw results separately
        raw_path = out_path.replace(".json", "_raw.jsonl")
        with open(raw_path, "w") as f:
            for r in results:
                # Make JSON-serializable
                r_clean = {k: v for k, v in r.items()
                           if isinstance(v, (str, int, float, list, dict, bool, type(None)))}
                f.write(json.dumps(r_clean) + "\n")
        print(f"Saved evaluation results to {out_path}")

    print(
        f"[{method_name}] accuracy={acc_info['accuracy']:.3f} "
        f"[{acc_info['ci_lower']:.3f}, {acc_info['ci_upper']:.3f}] "
        f"tokens/correct={eff:.1f} "
        f"wall={wall_clock:.1f}s"
    )
    return metrics
