"""Generate all result figures for InftyThink experiments."""
from __future__ import annotations
import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

FIGURES_DIR = "results/figures"


def _ensure_dir(path: str = FIGURES_DIR):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs. token budget (all methods)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_token_budget(
    method_results: dict[str, dict],
    out_path: Optional[str] = None,
):
    """Line/bar chart of accuracy for each method, with 95% CI error bars.

    Args:
        method_results: {"method_name": metrics_dict, ...}
    """
    _ensure_dir()
    names = list(method_results.keys())
    accs = [method_results[n]["accuracy"] for n in names]
    ci_lo = [method_results[n].get("accuracy_ci_lower", 0) for n in names]
    ci_hi = [method_results[n].get("accuracy_ci_upper", 0) for n in names]
    yerr_lo = [a - lo for a, lo in zip(accs, ci_lo)]
    yerr_hi = [hi - a for a, hi in zip(accs, ci_hi)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=PALETTE[:len(names)], edgecolor="white", linewidth=0.5)
    ax.errorbar(x, accs, yerr=[yerr_lo, yerr_hi], fmt="none", color="black",
                capsize=5, linewidth=1.5, label="95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, min(1.05, max(accs) * 1.3 + 0.05))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Final-Answer Accuracy by Method (with 95% CI)")
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig1_accuracy_by_method.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Token efficiency (tokens per correct answer)
# ---------------------------------------------------------------------------

def plot_token_efficiency(
    method_results: dict[str, dict],
    out_path: Optional[str] = None,
):
    _ensure_dir()
    names = list(method_results.keys())
    efficiencies = [method_results[n].get("token_efficiency", 0) for n in names]
    # Replace inf with a large sentinel for visualization
    max_finite = max((e for e in efficiencies if e != float("inf")), default=1)
    efficiencies_plot = [e if e != float("inf") else max_finite * 1.2 for e in efficiencies]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, efficiencies_plot, color=PALETTE[:len(names)], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mean tokens per correct answer (lower = better)")
    ax.set_title("Token Efficiency by Method")
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig2_token_efficiency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Peak context length (box/bar)
# ---------------------------------------------------------------------------

def plot_peak_context(
    method_results: dict[str, dict],
    out_path: Optional[str] = None,
):
    _ensure_dir()
    names = list(method_results.keys())
    means = [method_results[n].get("peak_context_mean", 0) for n in names]
    p95s  = [method_results[n].get("peak_context_p95", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, means, width, label="Mean", color=PALETTE[0])
    ax.bar(x + width / 2, p95s,  width, label="p95",  color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Context length (tokens)")
    ax.set_title("Peak Context Length by Method (Mean and p95)")
    ax.legend()
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig3_peak_context.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figures 4-6: Ablation line plots
# ---------------------------------------------------------------------------

def plot_ablation_line(
    ablation_results: dict[str, dict],
    x_label: str,
    x_vals: list,
    metric: str = "accuracy",
    title: str = "",
    out_path: Optional[str] = None,
    fig_idx: int = 4,
):
    """Line plot for a 1D ablation sweep."""
    _ensure_dir()
    y_vals = [ablation_results[str(x)].get(metric, 0) for x in x_vals]
    ci_lo = [ablation_results[str(x)].get("accuracy_ci_lower", y) for x in x_vals]
    ci_hi = [ablation_results[str(x)].get("accuracy_ci_upper", y) for x in x_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, color=PALETTE[0])
    ax.fill_between(range(len(x_vals)), ci_lo, ci_hi,
                    alpha=0.2, color=PALETTE[0], label="95% CI")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(x) for x in x_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric} vs. {x_label}")
    ax.legend()
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, f"fig{fig_idx}_ablation_{x_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_segment_length(ablation_results: dict, out_path: Optional[str] = None):
    plot_ablation_line(
        ablation_results, x_label="segment_len (tokens)",
        x_vals=[64, 128, 256], metric="accuracy",
        title="Accuracy vs. Segment Length (Ablation A1)",
        out_path=out_path, fig_idx=4,
    )


def plot_ablation_summary_length(ablation_results: dict, out_path: Optional[str] = None):
    plot_ablation_line(
        ablation_results, x_label="summary_len (tokens)",
        x_vals=[16, 32, 64], metric="accuracy",
        title="Accuracy vs. Summary Length (Ablation A2)",
        out_path=out_path, fig_idx=5,
    )


def plot_ablation_iterations(ablation_results: dict, out_path: Optional[str] = None):
    plot_ablation_line(
        ablation_results, x_label="T (iterations)",
        x_vals=[2, 4, 8], metric="accuracy",
        title="Accuracy vs. Number of Iterations (Ablation A3)",
        out_path=out_path, fig_idx=6,
    )


# ---------------------------------------------------------------------------
# Figure 7: Conditioning strategy (grouped bar)
# ---------------------------------------------------------------------------

def plot_conditioning_comparison(
    ablation_results: dict,
    out_path: Optional[str] = None,
):
    _ensure_dir()
    strategies = list(ablation_results.keys())
    accs = [ablation_results[s].get("accuracy", 0) for s in strategies]
    ci_lo = [ablation_results[s].get("accuracy_ci_lower", a) for s, a in zip(strategies, accs)]
    ci_hi = [ablation_results[s].get("accuracy_ci_upper", a) for s, a in zip(strategies, accs)]
    yerr_lo = [a - lo for a, lo in zip(accs, ci_lo)]
    yerr_hi = [hi - a for a, hi in zip(accs, ci_hi)]

    x = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, accs, color=PALETTE[:len(strategies)], edgecolor="white")
    ax.errorbar(x, accs, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Accuracy by Conditioning Strategy (Ablation A4)")
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig7_conditioning.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 8: Failure breakdown (stacked bar)
# ---------------------------------------------------------------------------

def plot_failure_breakdown(
    method_results: dict[str, dict],
    out_path: Optional[str] = None,
):
    _ensure_dir()
    from src.eval.failure_analyzer import FAILURE_CATEGORIES

    names = list(method_results.keys())
    cat_fractions = {cat: [] for cat in FAILURE_CATEGORIES}
    for name in names:
        fb = method_results[name].get("failure_breakdown", {})
        for cat in FAILURE_CATEGORIES:
            cat_fractions[cat].append(fb.get(cat, {}).get("fraction", 0))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(names))
    for i, cat in enumerate(FAILURE_CATEGORIES):
        vals = np.array(cat_fractions[cat])
        ax.bar(x, vals, bottom=bottoms, label=cat.replace("_", " "),
               color=PALETTE[i], edgecolor="white", linewidth=0.4)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Fraction of failures")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Failure Breakdown by Category and Method")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig8_failure_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 9: Structured vs. free-form summary (side-by-side with CI)
# ---------------------------------------------------------------------------

def plot_structured_vs_freeform(
    inftythink_result: dict,
    structured_result: dict,
    out_path: Optional[str] = None,
):
    _ensure_dir()
    methods = {
        "Free-form\n(InftyThink)": inftythink_result,
        "Structured\nState (E1)": structured_result,
    }
    names = list(methods.keys())
    accs = [methods[n]["accuracy"] for n in names]
    ci_lo = [methods[n].get("accuracy_ci_lower", a) for n, a in zip(names, accs)]
    ci_hi = [methods[n].get("accuracy_ci_upper", a) for n, a in zip(names, accs)]
    yerr = [[a - lo for a, lo in zip(accs, ci_lo)],
            [hi - a for a, hi in zip(accs, ci_hi)]]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(names))
    ax.bar(x, accs, color=[PALETTE[0], PALETTE[2]], edgecolor="white", linewidth=0.5, width=0.5)
    ax.errorbar(x, accs, yerr=yerr, fmt="none", color="black", capsize=8, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Free-form vs. Structured Summary (with 95% CI)")
    plt.tight_layout()
    path = out_path or os.path.join(FIGURES_DIR, "fig9_structured_vs_freeform.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Master function: generate all 9 figures
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str = "results"):
    """Load all result JSONs and generate all 9 figures."""
    import glob

    def _load(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # Method results
    b1 = _load(f"{results_dir}/b1_vanilla_cot.json")
    b2 = _load(f"{results_dir}/b2_capped_cot.json")
    b3 = _load(f"{results_dir}/b3_segmented_no_summary.json")
    b4 = _load(f"{results_dir}/b4_truncation.json")
    m1 = _load(f"{results_dir}/m1_inftythink.json")
    e1 = _load(f"{results_dir}/e1_structured_state.json")

    method_results = {}
    for name, r in [("B1 Vanilla CoT", b1), ("B2 Capped CoT", b2),
                     ("B3 No Summary", b3), ("B4 Truncation", b4),
                     ("M1 InftyThink", m1), ("E1 Structured", e1)]:
        if r is not None:
            method_results[name] = r

    if method_results:
        plot_accuracy_vs_token_budget(method_results)
        plot_token_efficiency(method_results)
        plot_peak_context(method_results)
        plot_failure_breakdown(method_results)

    # Ablation results
    a1 = _load(f"{results_dir}/ablations/a1_segment_length.json")
    a2 = _load(f"{results_dir}/ablations/a2_summary_length.json")
    a3 = _load(f"{results_dir}/ablations/a3_iterations.json")
    a4 = _load(f"{results_dir}/ablations/a4_conditioning.json")

    if a1:
        plot_ablation_segment_length(a1)
    if a2:
        plot_ablation_summary_length(a2)
    if a3:
        plot_ablation_iterations(a3)
    if a4:
        plot_conditioning_comparison(a4)

    if m1 and e1:
        plot_structured_vs_freeform(m1, e1)

    print("\nAll figures generated.")
