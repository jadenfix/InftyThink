"""Save and load model checkpoints using numpy serialization."""
from __future__ import annotations
import os
import json
import re
import shutil
import tempfile
import numpy as np
import jax
import jax.numpy as jnp


_CHECKPOINT_NAME = re.compile(r"^step_(\d{7})$")


def save_checkpoint(state, step: int, path: str = "checkpoints/") -> str:
    """Atomically save model parameters and metadata (not optimizer state).

    Files created:
        {path}/step_{step:07d}/params.npz
        {path}/step_{step:07d}/metadata.json

    Returns the checkpoint directory path.
    """
    os.makedirs(path, exist_ok=True)
    ckpt_dir = os.path.join(path, f"step_{step:07d}")
    if os.path.exists(ckpt_dir):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {ckpt_dir}")
    tmp_dir = tempfile.mkdtemp(prefix=f".step_{step:07d}.tmp-", dir=path)

    try:
        # Flatten params pytree to numpy arrays
        flat_params, treedef = jax.tree_util.tree_flatten(state.params)
        np_params = {str(i): np.array(p) for i, p in enumerate(flat_params)}
        params_path = os.path.join(tmp_dir, "params.npz")
        np.savez(params_path, **np_params)

        # Save metadata only after the parameter archive is complete.
        metadata = {
            "step": step,
            "treedef": str(treedef),
            "n_leaves": len(flat_params),
        }
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(params_path, "rb") as f:
            os.fsync(f.fileno())

        # A same-filesystem directory rename publishes the complete checkpoint
        # in one operation; hidden temporary directories are never discoverable.
        os.rename(tmp_dir, ckpt_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

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
    """Return complete, exactly named checkpoint directories sorted by step."""
    if not os.path.isdir(path):
        return []
    checkpoints: list[tuple[int, str]] = []
    for name in os.listdir(path):
        match = _CHECKPOINT_NAME.fullmatch(name)
        checkpoint = os.path.join(path, name)
        if not match or not os.path.isdir(checkpoint):
            continue
        params_path = os.path.join(checkpoint, "params.npz")
        metadata_path = os.path.join(checkpoint, "metadata.json")
        if not os.path.isfile(params_path) or not os.path.isfile(metadata_path):
            continue
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            if metadata.get("step") != int(match.group(1)):
                continue
            n_leaves = metadata.get("n_leaves")
            if not isinstance(n_leaves, int) or n_leaves <= 0:
                continue
            with np.load(params_path) as archive:
                if set(archive.files) != {str(index) for index in range(n_leaves)}:
                    continue
                for index in range(n_leaves):
                    np.asarray(archive[str(index)])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        checkpoints.append((int(match.group(1)), checkpoint))
    return [checkpoint for _, checkpoint in sorted(checkpoints)]


def latest_checkpoint(path: str = "checkpoints/") -> str | None:
    """Return the path of the most recent checkpoint, or None."""
    ckpts = list_checkpoints(path)
    return ckpts[-1] if ckpts else None
