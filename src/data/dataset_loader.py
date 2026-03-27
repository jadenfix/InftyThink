"""Load and subset OpenR1-Math-220k from HuggingFace."""
import re
from datasets import load_dataset, Dataset


def load_openr1(
    n_train: int = 5000,
    n_eval: int = 500,
    seed: int = 42,
    cache_dir: str = None,
) -> dict[str, Dataset]:
    """Load open-r1/OpenR1-Math-220k default split.

    Returns {"train": Dataset, "eval": Dataset} with columns:
        problem (str), solution (str), answer (str)

    Filters to examples that contain a <think>...</think> block.
    Train and eval subsets are non-overlapping.
    """
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        "default",
        split="train",
        cache_dir=cache_dir,
    )

    # Normalize column names: dataset uses "problem" and "solution"
    required = {"problem", "solution", "answer"}
    available = set(ds.column_names)
    missing = required - available
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Available: {available}")

    # Filter: keep only examples with <think>...</think>
    def has_think_block(example):
        return bool(re.search(r"<think>.*?</think>", example["solution"], re.DOTALL))

    ds = ds.filter(has_think_block, num_proc=4)

    # Shuffle deterministically
    ds = ds.shuffle(seed=seed)

    total_needed = n_train + n_eval
    if len(ds) < total_needed:
        raise ValueError(
            f"Only {len(ds)} examples after filtering, need {total_needed}"
        )

    train_ds = ds.select(range(n_train))
    eval_ds = ds.select(range(n_train, n_train + n_eval))

    return {"train": train_ds, "eval": eval_ds}


def extract_think_content(solution: str) -> str:
    """Extract the reasoning trace from inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return solution as-is
    return solution.strip()


def extract_answer(solution: str) -> str:
    """Extract the answer portion after </think>, or the raw answer field."""
    after_think = re.sub(r"<think>.*?</think>", "", solution, flags=re.DOTALL).strip()
    if after_think:
        return after_think
    return solution.strip()
