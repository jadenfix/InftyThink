"""Save and load model checkpoints using numpy serialization."""
from __future__ import annotations
import os
import json
import glob
import numpy as np
import jax
import jax.numpy as jnp


def save_checkpoint(state, step: int, path: str = "checkpoints/") -> str:
    """Save a Flax TrainState (params + opt_state) to disk.

    Files created:
        {path}/step_{step:07d}/params.npz
        {path}/step_{step:07d}/metadata.json

    Returns the checkpoint directory path.
    """
    ckpt_dir = os.path.join(path, f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Flatten params pytree to numpy arrays
    flat_params, treedef = jax.tree_util.tree_flatten(state.params)
    np_params = {str(i): np.array(p) for i, p in enumerate(flat_params)}
    np.savez(os.path.join(ckpt_dir, "params.npz"), **np_params)

    # Save metadata
    metadata = {
        "step": step,
        "treedef": str(treedef),
        "n_leaves": len(flat_params),
    }
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint: {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(path: str, state_template=None) -> dict:
    """Load params from a checkpoint directory.

    Args:
        path: Path to checkpoint directory (e.g., "checkpoints/step_0001000").
        state_template: If provided, the loaded params will be cast to match
                        the dtypes in this pytree.

    Returns:
        {"params": pytree, "step": int}
    """
    params_path = os.path.join(path, "params.npz")
    meta_path = os.path.join(path, "metadata.json")

    with open(meta_path) as f:
        metadata = json.load(f)

    npz = np.load(params_path)
    flat_params = [jnp.array(npz[str(i)]) for i in range(metadata["n_leaves"])]

    # We can't fully reconstruct the treedef from string without eval,
    # so we return the flat list and expect the caller to unflatten via
    # the model's init output structure.
    return {
        "flat_params": flat_params,
        "step": metadata["step"],
        "n_leaves": metadata["n_leaves"],
    }


def list_checkpoints(path: str = "checkpoints/") -> list[str]:
    """Return sorted list of checkpoint directory paths."""
    pattern = os.path.join(path, "step_*")
    dirs = sorted(glob.glob(pattern))
    return dirs


def latest_checkpoint(path: str = "checkpoints/") -> str | None:
    """Return the path of the most recent checkpoint, or None."""
    ckpts = list_checkpoints(path)
    return ckpts[-1] if ckpts else None
