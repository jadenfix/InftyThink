"""Compute and save quantitative statistics on the dataset."""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.data.dataset_loader import extract_think_content
from src.data.segmenter import segment_trace, compute_trace_token_length


def compute_and_save_stats(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    out_path: str = "results/dataset_stats.json",
    figures_dir: str = "results/figures",
    segment_lengths: list[int] = (64, 128, 256),
) -> dict:
    """Compute quantitative statistics over the dataset and save to disk.

    Metrics computed:
      - Raw trace token length distribution (p25/50/75/95/99)
      - Mean segments-per-example at each segment_len setting
      - Summary compression ratio: summary_tokens / segment_tokens
      - Answer type distribution (numeric / expression / other)

    Saves:
      - out_path: JSON with all statistics
      - figures_dir/token_length_hist.png
      - figures_dir/segments_per_example.png
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    traces = [extract_think_content(row["solution"]) for row in dataset]
    trace_lengths = np.array(
        [compute_trace_token_length(t, tokenizer) for t in traces], dtype=np.float32
    )

    # --- token length percentiles ---
    percentiles = {
        f"p{p}": float(np.percentile(trace_lengths, p))
        for p in (25, 50, 75, 95, 99)
    }
    percentiles["mean"] = float(np.mean(trace_lengths))
    percentiles["std"] = float(np.std(trace_lengths))
    percentiles["min"] = float(np.min(trace_lengths))
    percentiles["max"] = float(np.max(trace_lengths))

    # --- segments per example at each segment_len ---
    segments_per_example: dict[str, float] = {}
    for seg_len in segment_lengths:
        n_segs = []
        for trace in traces:
            segs = segment_trace(trace, tokenizer, segment_len=seg_len)
            n_segs.append(len(segs))
        arr = np.array(n_segs, dtype=np.float32)
        segments_per_example[str(seg_len)] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    # --- answer type distribution ---
    answer_types = _classify_answers([row["answer"] for row in dataset])

    stats = {
        "n_examples": len(dataset),
        "trace_token_lengths": percentiles,
        "segments_per_example": segments_per_example,
        "answer_type_distribution": answer_types,
    }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset stats to {out_path}")

    # --- plots ---
    _plot_token_length_hist(trace_lengths, figures_dir)
    _plot_segments_per_example(segments_per_example, figures_dir)

    return stats


def _classify_answers(answers: list[str]) -> dict[str, float]:
    """Classify answers as numeric, fraction, expression, or other."""
    import re
    counts = {"numeric": 0, "fraction": 0, "expression": 0, "other": 0}
    for ans in answers:
        a = ans.strip()
        if re.fullmatch(r"-?[\d,]+\.?\d*", a.replace(",", "")):
            counts["numeric"] += 1
        elif re.fullmatch(r"-?[\d]+/[\d]+", a):
            counts["fraction"] += 1
        elif re.search(r"[a-zA-Z]", a):
            counts["expression"] += 1
        else:
            counts["other"] += 1
    total = max(len(answers), 1)
    return {k: round(v / total, 4) for k, v in counts.items()}


def _plot_token_length_hist(lengths: np.ndarray, figures_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, edgecolor="white", linewidth=0.4)
    ax.axvline(np.percentile(lengths, 50), color="red", linestyle="--", label="p50")
    ax.axvline(np.percentile(lengths, 95), color="orange", linestyle="--", label="p95")
    ax.set_xlabel("Trace token length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of reasoning trace token lengths")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(figures_dir, "token_length_hist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_segments_per_example(
    segments_per_example: dict[str, dict], figures_dir: str
):
    seg_lens = sorted(segments_per_example.keys(), key=int)
    means = [segments_per_example[k]["mean"] for k in seg_lens]
    stds = [segments_per_example[k]["std"] for k in seg_lens]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(seg_lens, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xlabel("Segment length (tokens)")
    ax.set_ylabel("Mean segments per example")
    ax.set_title("Mean segments per example vs. segment length")
    plt.tight_layout()
    path = os.path.join(figures_dir, "segments_per_example.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
