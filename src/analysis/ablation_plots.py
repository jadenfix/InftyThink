"""2D ablation heatmaps for joint parameter sweeps."""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "results/figures"


def plot_ablation_heatmap(
    results_grid: dict,
    x_param: str,
    y_param: str,
    metric: str = "accuracy",
    x_vals: list | None = None,
    y_vals: list | None = None,
    out_path: str | None = None,
    title: str | None = None,
    fmt: str = ".3f",
):
    """2D heatmap of a metric over a grid of two hyperparameters.

    Args:
        results_grid: nested dict {x_val: {y_val: metrics_dict}}
        x_param: name of the x-axis parameter (e.g., "segment_len")
        y_param: name of the y-axis parameter (e.g., "summary_len")
        metric: metric key to visualize (e.g., "accuracy")
        x_vals: ordered list of x-axis values (inferred if None)
        y_vals: ordered list of y-axis values (inferred if None)
        out_path: save path (auto-generated if None)
        title: plot title
        fmt: cell annotation format

    Example results_grid structure:
        {"64": {"16": {"accuracy": 0.32}, "32": {"accuracy": 0.35}, ...}, "128": {...}}
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if x_vals is None:
        x_vals = sorted(results_grid.keys(), key=lambda k: float(k))
    if y_vals is None:
        first_x = str(x_vals[0])
        y_vals = sorted(results_grid[first_x].keys(), key=lambda k: float(k))

    matrix = np.zeros((len(y_vals), len(x_vals)))
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            val = results_grid.get(str(xv), {}).get(str(yv), {}).get(metric, float("nan"))
            matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        xticklabels=[str(v) for v in x_vals],
        yticklabels=[str(v) for v in y_vals],
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title or f"{metric} over {x_param} × {y_param}")
    plt.tight_layout()

    path = out_path or os.path.join(
        FIGURES_DIR, f"heatmap_{x_param}_x_{y_param}_{metric}.png"
    )
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def build_2d_grid_from_runs(
    a1_results: dict,
    a2_results: dict,
) -> dict:
    """Construct a 2D results grid from two independent 1D ablation results.

    This approximates a joint sweep by combining A1 (segment_len) and
    A2 (summary_len) results, using M1 defaults for the non-swept parameter.

    Returns nested dict: {segment_len: {summary_len: metrics_dict}}
    """
    grid = {}
    for seg_len, seg_metrics in a1_results.items():
        grid[seg_len] = {}
        for sum_len, sum_metrics in a2_results.items():
            # Use accuracy from A1 for this segment_len as the primary value
            # This is an approximation; a true joint sweep would be preferred.
            grid[seg_len][sum_len] = {
                "accuracy": (seg_metrics.get("accuracy", 0) + sum_metrics.get("accuracy", 0)) / 2
            }
    return grid
