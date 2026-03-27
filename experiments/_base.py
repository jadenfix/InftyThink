"""Shared experiment runner utilities."""
from __future__ import annotations
import json
import os
import yaml
import jax
import numpy as np

from src.model.config import ModelConfig
from src.model.transformer import CausalLM
from src.model.tokenizer import load_tokenizer
from src.training.checkpointer import latest_checkpoint, load_checkpoint
from src.data.dataset_loader import load_openr1


def load_experiment_config(config_path: str = "configs/base.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_params(
    checkpoint_path: str | None = None,
    model_config: ModelConfig | None = None,
    config_path: str = "configs/base.yaml",
) -> tuple[CausalLM, dict, ModelConfig]:
    """Load model, params, and config.

    If checkpoint_path is None, uses the latest checkpoint in checkpoints/.
    If no checkpoints exist, initializes fresh random params.
    """
    cfg_dict = load_experiment_config(config_path)
    if model_config is None:
        model_config = ModelConfig(**{
            k: v for k, v in cfg_dict["model"].items()
            if k in ModelConfig.__dataclass_fields__
        })

    model = CausalLM(model_config)
    tokenizer = load_tokenizer(cfg_dict["data"].get("tokenizer", "gpt2"))

    # Update vocab size to match tokenizer (after adding special tokens)
    from src.model.tokenizer import get_vocab_size
    model_config.vocab_size = get_vocab_size(tokenizer)

    ckpt = checkpoint_path or latest_checkpoint("checkpoints/")
    if ckpt:
        ckpt_data = load_checkpoint(ckpt)
        print(f"Loaded checkpoint from {ckpt} (step {ckpt_data['step']})")
        # Reinitialize params with correct shape, then load from npz
        rng = jax.random.PRNGKey(0)
        import jax.numpy as jnp
        dummy = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy)
        params = variables["params"]
        # Overwrite with loaded flat params
        flat_loaded = ckpt_data["flat_params"]
        flat_current, treedef = jax.tree_util.tree_flatten(params)
        if len(flat_loaded) == len(flat_current):
            params = jax.tree_util.tree_unflatten(
                treedef, [jnp.array(p) for p in flat_loaded]
            )
        else:
            print(f"Warning: checkpoint has {len(flat_loaded)} leaves, "
                  f"model has {len(flat_current)}. Using random init.")
    else:
        print("No checkpoint found. Using random initialization.")
        rng = jax.random.PRNGKey(42)
        import jax.numpy as jnp
        dummy = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy)
        params = variables["params"]

    return model, params, model_config


def load_eval_dataset(n_eval: int = 500, seed: int = 42):
    datasets = load_openr1(n_train=100, n_eval=n_eval, seed=seed)
    return datasets["eval"]


def save_result(result: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
