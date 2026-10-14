"""InftyThink — main CLI entry point.

Usage:
    python main.py prepare-data   [--n-train N] [--n-eval N] [--tokenizer gpt2]
    python main.py train          [--config configs/base.yaml]
    python main.py evaluate       [--method METHOD] [--checkpoint PATH] [--n-eval N]
    python main.py run-experiment [--name EXPERIMENT_NAME]
    python main.py analyze        [--results-dir results/]
    python main.py stats          [--n N]
"""
from __future__ import annotations
import argparse
import os
import sys

# Auto-detect: only force CPU if no GPU/TPU is available
if "JAX_PLATFORMS" not in os.environ:
    try:
        import jax
        if not any("gpu" in str(d).lower() or "tpu" in str(d).lower() for d in jax.devices()):
            os.environ["JAX_PLATFORMS"] = "cpu"
    except Exception:
        os.environ["JAX_PLATFORMS"] = "cpu"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_prepare_data(args):
    from src.data.dataset_loader import load_openr1
    from src.data.data_converter import convert_example
    from src.data.dataset_stats import compute_and_save_stats
    from src.model.tokenizer import load_tokenizer

    print(f"Loading OpenR1-Math-220k (n_train={args.n_train}, n_eval={args.n_eval}) ...")
    datasets = load_openr1(n_train=args.n_train, n_eval=args.n_eval, seed=args.seed)
    tokenizer = load_tokenizer(args.tokenizer)

    print("Computing dataset statistics ...")
    compute_and_save_stats(
        datasets["train"],
        tokenizer,
        out_path="results/dataset_stats.json",
        figures_dir="results/figures",
        segment_lengths=[64, 128, 256],
    )
    print(f"Train: {len(datasets['train'])} examples")
    print(f"Eval:  {len(datasets['eval'])} examples")
    print("Done.")


def cmd_train(args):
    if args.checkpoint:
        raise ValueError(
            "Training checkpoint resume is not implemented: checkpoints contain "
            "parameters but not optimizer state. Refusing to restart or overwrite."
        )
    import jax
    import yaml
    from src.model.config import ModelConfig
    from src.model.transformer import CausalLM
    from src.model.tokenizer import load_tokenizer, get_vocab_size
    from src.data.dataset_loader import load_openr1, extract_think_content
    from src.data.data_converter import convert_example
    from src.training.trainer import train, TrainConfig

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_config = ModelConfig(**{
        k: v for k, v in cfg["model"].items()
        if k in ModelConfig.__dataclass_fields__
    })
    train_cfg = TrainConfig(**{
        k: v for k, v in cfg["training"].items()
        if k in TrainConfig.__dataclass_fields__
    })

    tokenizer = load_tokenizer(cfg["data"].get("tokenizer", "gpt2"))
    model_config.vocab_size = get_vocab_size(tokenizer)
    model = CausalLM(model_config)

    print("Loading dataset ...")
    datasets = load_openr1(
        n_train=cfg["data"]["n_train"],
        n_eval=cfg["data"]["n_eval"],
        seed=cfg["data"]["seed"],
    )

    seg_len = cfg["data"]["segment_len"]
    sum_len = cfg["data"]["summary_len"]

    print(f"Converting training data (segment_len={seg_len}, summary_len={sum_len}) ...")
    train_instances = []
    for row in datasets["train"]:
        trace = extract_think_content(row["solution"])
        instances = convert_example(
            problem=row["problem"],
            trace=trace,
            answer=row["answer"],
            tokenizer=tokenizer,
            segment_len=seg_len,
            summary_len=sum_len,
            max_seq_len=model_config.max_seq_len,
        )
        train_instances.extend(instances)

    eval_instances = []
    for row in datasets["eval"]:
        trace = extract_think_content(row["solution"])
        instances = convert_example(
            problem=row["problem"],
            trace=trace,
            answer=row["answer"],
            tokenizer=tokenizer,
            segment_len=seg_len,
            summary_len=sum_len,
            max_seq_len=model_config.max_seq_len,
        )
        eval_instances.extend(_select_training_eval_instances(instances))

    print(f"Train instances: {len(train_instances)}, Eval instances: {len(eval_instances)}")

    train(model, model_config, train_instances, eval_instances, train_cfg)


def cmd_evaluate(args):
    from functools import partial
    from src.model.tokenizer import load_tokenizer
    from src.eval.evaluator import evaluate
    from experiments._base import load_eval_dataset, load_model_and_params

    model, params, model_config = load_model_and_params(
        checkpoint_path=args.checkpoint
    )
    tokenizer = load_tokenizer()
    eval_ds = load_eval_dataset(n_eval=args.n_eval)

    METHOD_MAP = {
        "vanilla_cot": _make_vanilla_cot_fn,
        "capped_cot": _make_capped_cot_fn,
        "inftythink": _make_inftythink_fn,
        "structured_state": _make_structured_fn,
    }
    if args.method not in METHOD_MAP:
        print(f"Unknown method: {args.method}. Choose from: {list(METHOD_MAP.keys())}")
        sys.exit(1)

    inference_fn = METHOD_MAP[args.method]()
    out_path = f"results/{args.method}_eval.json"

    evaluate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        inference_fn=inference_fn,
        n_examples=args.n_eval,
        out_path=out_path,
        method_name=args.method,
    )


def _select_training_eval_instances(instances: list[dict], limit: int = 4) -> list[dict]:
    """Cap converted eval instances while retaining the final-answer task."""
    if len(instances) <= limit:
        return instances
    selected = instances[:limit - 1]
    final = instances[-1]
    if final.get("task") != "final":
        raise ValueError("Converted example must end with a final task")
    return [*selected, final]


def _make_vanilla_cot_fn():
    from functools import partial
    from src.inference.vanilla_cot import generate_vanilla_cot
    return partial(generate_vanilla_cot, max_tokens=2048)


def _make_capped_cot_fn():
    from functools import partial
    from src.inference.vanilla_cot import generate_vanilla_cot
    return partial(generate_vanilla_cot, max_tokens=768)


def _make_inftythink_fn():
    from functools import partial
    from src.inference.iterative_reasoner import run_iterative_reasoning, IterativeConfig
    config = IterativeConfig(segment_len=128, summary_len=32, max_iterations=4)
    return partial(run_iterative_reasoning, config=config)


def _make_structured_fn():
    from functools import partial
    from src.inference.structured_state import run_structured_iterative
    from src.inference.iterative_reasoner import IterativeConfig
    config = IterativeConfig(segment_len=128, summary_len=32, max_iterations=4)
    return partial(run_structured_iterative, config=config)


def cmd_run_experiment(args):
    EXPERIMENTS = {
        "b1_vanilla_cot":          "experiments.baseline_vanilla_cot",
        "b2_capped_cot":           "experiments.baseline_capped_cot",
        "b3_segmented_no_summary": "experiments.baseline_segmented_no_summary",
        "b4_truncation":           "experiments.baseline_truncation",
        "m1_inftythink":           "experiments.run_inftythink",
        "a1_segment_length":       "experiments.ablation_segment_length",
        "a2_summary_length":       "experiments.ablation_summary_length",
        "a3_iterations":           "experiments.ablation_iterations",
        "a4_conditioning":         "experiments.ablation_conditioning",
        "e1_structured_state":     "experiments.extension_structured_state",
    }
    if args.name not in EXPERIMENTS:
        print(f"Unknown experiment: {args.name}. Choose from:\n  " + "\n  ".join(EXPERIMENTS))
        sys.exit(1)

    import importlib
    mod = importlib.import_module(EXPERIMENTS[args.name])
    mod.run()


def cmd_analyze(args):
    from src.analysis.plot_results import generate_all_figures

    generate_all_figures(args.results_dir)


def cmd_stats(args):
    """Quick tokenizer and model statistics check."""
    from src.model.config import ModelConfig
    from src.model.tokenizer import load_tokenizer, get_vocab_size

    cfg = ModelConfig()
    tokenizer = load_tokenizer()
    vocab = get_vocab_size(tokenizer)
    cfg.vocab_size = vocab

    print(f"Tokenizer vocab size (with special tokens): {vocab}")
    print(f"Model config: {cfg.to_dict()}")
    print(f"Approx param count: {cfg.param_count_estimate():,}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InftyThink: Iterative Reasoning with JAX on CPU"
    )
    sub = parser.add_subparsers(dest="command")

    # prepare-data
    p = sub.add_parser("prepare-data", help="Download and preprocess dataset")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", type=str, default="gpt2")

    # train
    p = sub.add_parser("train", help="Train the model")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="unsupported for training; fails closed because optimizer state is not stored",
    )

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate a method on the eval set")
    p.add_argument(
        "--method", type=str, default="inftythink",
        choices=["vanilla_cot", "capped_cot", "inftythink", "structured_state"],
    )
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n-eval", type=int, default=500)

    # run-experiment
    p = sub.add_parser("run-experiment", help="Run a named experiment using configs/base.yaml")
    p.add_argument("--name", type=str, required=True)

    # analyze
    p = sub.add_parser("analyze", help="Generate all figures from saved results")
    p.add_argument("--results-dir", type=str, default="results")

    # stats
    sub.add_parser("stats", help="Print model and tokenizer statistics")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "prepare-data": cmd_prepare_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "run-experiment": cmd_run_experiment,
        "analyze": cmd_analyze,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
