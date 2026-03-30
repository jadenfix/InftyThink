"""Load and subset OpenR1-Math-220k from HuggingFace."""
import re
from datasets import load_dataset, Dataset


def _pick_correct_generation(example: dict) -> str | None:
    """Return the first correct generation containing <think>...</think>.

    The dataset stores reasoning traces in `generations` (list[str]) with
    per-generation correctness in `correctness_math_verify` (list[bool]).
    We pick the first generation that is both correct and has a think block.
    Falls back to the first generation with a think block if none marked correct.
    """
    gens = example.get("generations", [])
    correct = example.get("correctness_math_verify", [])

    # First pass: correct generation with think block
    for i, g in enumerate(gens):
        is_correct = correct[i] if i < len(correct) else False
        if is_correct and re.search(r"<think>", str(g)):
            return str(g)

    # Second pass: any generation with think block
    for g in gens:
        if re.search(r"<think>", str(g)):
            return str(g)

    return None


def load_openr1(
    n_train: int = 5000,
    n_eval: int = 500,
    seed: int = 42,
    cache_dir: str = None,
) -> dict[str, Dataset]:
    """Load open-r1/OpenR1-Math-220k default split.

    Returns {"train": Dataset, "eval": Dataset} with columns:
        problem (str), solution (str), answer (str)

    The reasoning traces come from the `generations` column (which contains
    <think>...</think> blocks), not the `solution` column. We pick the first
    correct generation per example and store it as `solution`.

    Filters to examples that have at least one generation with a <think> block.
    Train and eval subsets are non-overlapping.
    """
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        "default",
        split="train",
        cache_dir=cache_dir,
    )

    required = {"problem", "answer", "generations"}
    available = set(ds.column_names)
    missing = required - available
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Available: {available}")

    # Map: pick the best generation and store as "solution"
    def extract_solution(example):
        gen = _pick_correct_generation(example)
        example["solution"] = gen if gen is not None else ""
        return example

    ds = ds.map(extract_solution, num_proc=4)

    # Filter: keep only examples where we found a think block
    ds = ds.filter(lambda ex: bool(ex["solution"]), num_proc=4)

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
