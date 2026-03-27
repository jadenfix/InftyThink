# InftyThink

A JAX/CPU reproduction and extension of [InftyThink (arxiv:2503.06692)](https://arxiv.org/abs/2503.06692) — iterative reasoning with bounded segments and progress summaries on mathematical problem solving.

**Core claim**: generating reasoning in short segments with a compressed summary at each step matches or outperforms vanilla chain-of-thought at equal token budget, while keeping peak context length bounded.

---

## How It Works

InftyThink replaces a single long chain-of-thought with an iterative loop:

```
for t in 1..T:
    generate segment (K tokens) | [question, summary_1..summary_{t-1}]
    generate summary (S tokens) | [question, summary_1..summary_{t-1}, segment_t]
generate final answer           | [question, summary_1..summary_T]
```

- **Segment** (`K=128` tokens): the model reasons freely for a bounded window
- **Summary** (`S=32` tokens): the model compresses its progress into a key-facts digest
- **Context** at each step: question + all prior summaries (never the full trace)
- **Peak context** is bounded by `len(question) + T*S` regardless of reasoning depth

This is compared against 4 baselines under the same 2048-token total budget.

---

## Environment Setup

Requires [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
bash scripts/setup_env.sh
```

This will:
1. Install `uv` if not present
2. Create `.venv` with `uv venv`
3. Install all pinned dependencies with `uv pip install -r requirements.txt`
4. Verify JAX CPU backend (`[CpuDevice]`)

All experiments run on **CPU only** (`JAX_PLATFORMS=cpu`). No GPU required.

**Key dependencies** (pinned in `requirements.txt`):
| Package | Version |
|---------|---------|
| jax[cpu] | 0.4.28 |
| flax | 0.8.4 |
| optax | 0.2.3 |
| transformers | 4.40.0 |
| datasets | 2.19.0 |
| numpy | 1.26.4 |
| scipy | 1.13.0 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |

---

## Quick Start

```bash
# 1. Set up environment
bash scripts/setup_env.sh

# 2. Download and preprocess data (5k train / 500 eval from OpenR1-Math-220k)
bash scripts/prepare_data.sh

# 3. Train the model (~60M params, 10k steps)
source .venv/bin/activate
python main.py train --config configs/base.yaml

# 4. Run all experiments
bash scripts/run_experiments.sh

# 5. Generate all figures
python main.py analyze --results-dir results/
```

---

## Model Architecture

Decoder-only transformer, CPU-feasible at ~60M parameters.

| Hyperparameter | Value |
|---------------|-------|
| Layers | 6 |
| d_model | 512 |
| Attention heads | 8 (head_dim=64) |
| d_ff | 2048 (SwiGLU) |
| Max seq len | 1024 |
| Vocab size | 32,000 (GPT-2) |
| Positional encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Tied embeddings | Yes |

---

## Training

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 8 (effective 32 w/ grad accum ×4) |
| Learning rate | 3e-4 |
| LR schedule | Cosine w/ linear warmup (200 steps) |
| Total steps | 10,000 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Optimizer | AdamW |

Training data: **5,000 examples** from `open-r1/OpenR1-Math-220k`, filtered to those containing `<think>...</think>` reasoning traces. Each example is segmented and converted into (segment, summary, final) training instances.

```bash
python main.py train --config configs/base.yaml
# Checkpoints saved to checkpoints/ every 1,000 steps
# Training log saved to checkpoints/train_log.jsonl
```

---

## Experiments

### Baselines

| ID | Method | Description |
|----|--------|-------------|
| B1 | Vanilla CoT | Uncapped chain-of-thought (2048 token budget) |
| B2 | Capped CoT | Chain-of-thought capped at same effective budget as InftyThink |
| B3 | Segmented, No Summary | Segments without compression; full context passed |
| B4 | Truncation | Only latest segment as context; no memory |

### Main Method

| ID | Method | Config |
|----|--------|--------|
| M1 | InftyThink | segment_len=128, summary_len=32, T=4, summary_only |

### Ablations

| ID | What varies | Values |
|----|-------------|--------|
| A1 | Segment length K | {64, 128, 256} tokens |
| A2 | Summary length S | {16, 32, 64} tokens |
| A3 | Iterations T | {2, 4, 8} |
| A4 | Conditioning strategy | summary_only / summary+tail / rolling_state |

### Extension

| ID | Method | Description |
|----|--------|-------------|
| E1 | Structured State | Structured `{known_facts, open_subgoals, derived_values, constraints, confidence}` schema instead of free-form summaries |

Run all experiments in dependency order:

```bash
bash scripts/run_experiments.sh
```

Or run individually:

```bash
python main.py run-experiment --name m1_inftythink
python main.py run-experiment --name a1_segment_length
# Valid names: b1_vanilla_cot, b2_capped_cot, b3_segmented_no_summary, b4_truncation,
#              m1_inftythink, a1_segment_length, a2_summary_length, a3_iterations,
#              a4_conditioning, e1_structured_state
```

Results are saved to `results/*.json` (baselines/main) and `results/ablations/*.json`.

---

## Evaluation

All methods are evaluated on **500 held-out examples** under a shared **2048-token budget**.

Metrics reported:
- **Accuracy** — exact final-answer match (numeric tolerance 1e-6, fraction equivalence)
- **Token efficiency** — mean tokens per correct answer (lower = better)
- **Compression ratio** — mean `summary_tokens / segment_tokens` per step
- **Peak context length** — mean and p95 over eval examples
- **95% bootstrap CI** — 10,000 bootstrap samples over accuracy
- **McNemar's test** — pairwise significance between M1 and each baseline

Failure analysis classifies wrong answers into 5 categories:
- `constraint_loss` — problem constraints absent from summary
- `numeric_drift` — numeric values mutated across steps
- `subgoal_omission` — unfinished subgoals not carried forward
- `overcompression` — summary too vague to support continuation
- `redundant_overhead` — budget wasted on trivial progress

```bash
python main.py evaluate --method inftythink --n-eval 500
# Output: results/inftythink_eval.json
```

---

## Figures

Nine figures are auto-generated from result JSONs:

| Figure | Content |
|--------|---------|
| fig1 | Accuracy by method (bar + 95% CI) |
| fig2 | Token efficiency by method |
| fig3 | Peak context length mean + p95 |
| fig4 | Accuracy vs. segment length (A1) |
| fig5 | Accuracy vs. summary length (A2) |
| fig6 | Accuracy vs. iterations T (A3) |
| fig7 | Conditioning strategy comparison (A4) |
| fig8 | Failure breakdown stacked bar |
| fig9 | Structured vs. free-form summary (M1 vs E1) |

```bash
python main.py analyze --results-dir results/
# Figures saved to results/figures/
```

---

## Project Structure

```
reasoning/
├── configs/
│   └── base.yaml              # All hyperparameters
├── experiments/
│   ├── _base.py               # Shared utilities
│   ├── baseline_vanilla_cot.py
│   ├── baseline_capped_cot.py
│   ├── baseline_segmented_no_summary.py
│   ├── baseline_truncation.py
│   ├── run_inftythink.py
│   ├── ablation_segment_length.py
│   ├── ablation_summary_length.py
│   ├── ablation_iterations.py
│   ├── ablation_conditioning.py
│   └── extension_structured_state.py
├── src/
│   ├── data/
│   │   ├── dataset_loader.py  # OpenR1-Math-220k loading
│   │   ├── segmenter.py       # Token-bounded trace segmentation
│   │   ├── summary_generator.py  # Heuristic + structured summaries
│   │   ├── data_converter.py  # Training instance construction
│   │   └── dataset_stats.py   # Token length / compression statistics
│   ├── model/
│   │   ├── config.py          # ModelConfig dataclass
│   │   ├── transformer.py     # RoPE + RMSNorm + SwiGLU decoder
│   │   └── tokenizer.py       # GPT-2 tokenizer + control tokens
│   ├── training/
│   │   ├── losses.py          # Masked CE + segment/summary split loss
│   │   ├── lr_schedule.py     # Cosine schedule with warmup
│   │   ├── checkpointer.py    # .npz checkpoint save/load
│   │   └── trainer.py         # JIT train_step, full training loop
│   ├── inference/
│   │   ├── generation_utils.py  # Greedy / nucleus sampling, answer extraction
│   │   ├── vanilla_cot.py     # B1/B2 baseline inference
│   │   ├── iterative_reasoner.py  # InftyThink core loop
│   │   └── structured_state.py   # E1 structured state extension
│   ├── eval/
│   │   ├── answer_extractor.py  # Normalize + match answers
│   │   ├── metrics.py           # Accuracy, efficiency, bootstrap CI
│   │   ├── failure_analyzer.py  # 5-category failure classification
│   │   └── evaluator.py         # Full eval runner
│   └── analysis/
│       ├── statistical_tests.py  # McNemar, paired bootstrap
│       ├── plot_results.py        # 9 figures
│       └── ablation_plots.py      # 2D heatmap for joint sweeps
├── scripts/
│   ├── setup_env.sh           # uv environment setup
│   ├── prepare_data.sh        # Data preprocessing
│   └── run_experiments.sh     # Full experiment suite
├── main.py                    # CLI entry point
├── requirements.txt           # Pinned dependencies
└── results/                   # Output JSONs and figures (git-ignored)
```

---

## CLI Reference

```bash
python main.py prepare-data  --n-train 5000 --n-eval 500 --seed 42 --tokenizer gpt2
python main.py train         --config configs/base.yaml [--checkpoint PATH]
python main.py evaluate      --method inftythink --n-eval 500
python main.py run-experiment --name m1_inftythink
python main.py analyze       --results-dir results/
python main.py stats         # Print model/tokenizer parameter counts
```

---

## Reference

> *InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models*
> arxiv:2503.06692
