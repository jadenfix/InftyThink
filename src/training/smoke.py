"""Download-free tiny model training/checkpoint/generation computational smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.model.config import ModelConfig
from src.model.transformer import CausalLM
from src.training.checkpointer import load_checkpoint, save_checkpoint
from src.training.trainer import TrainConfig, create_train_state, train_step


def run_smoke(output: Path) -> dict:
    model_config = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2,
                               d_ff=32, max_seq_len=16, dropout=0.0)
    config = TrainConfig(max_steps=2, warmup_steps=0, learning_rate=.001)
    model = CausalLM(model_config)
    state = create_train_state(model, config, model_config, jax.random.PRNGKey(7))
    before = jax.tree.map(lambda leaf: leaf.copy(), state.params)
    ids = jnp.asarray(np.arange(24).reshape(3, 8) % 32, dtype=jnp.int32)
    batch = {"input_ids": ids, "target_ids": (ids + 1) % 32,
             "loss_mask": jnp.ones(ids.shape), "task_flags": jnp.asarray([0, 1, 2])}
    losses = []
    for step in range(2):
        state, metrics = train_step(state, [batch], jax.random.PRNGKey(step))
        losses.append(metrics["loss"])
    delta = sum(float(jnp.sum(jnp.abs(a - b)))
                for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(state.params), strict=True))
    if not all(np.isfinite(loss) for loss in losses) or not delta > 0:
        raise RuntimeError("Tiny training produced invalid loss or unchanged parameters")
    checkpoint = save_checkpoint(state, int(state.step), str(output / "checkpoints"))
    loaded = load_checkpoint(checkpoint)
    leaves, structure = jax.tree.flatten(state.params)
    if len(leaves) != loaded["n_leaves"]:
        raise RuntimeError("Checkpoint leaf count changed")
    for expected, actual in zip(leaves, loaded["flat_params"], strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    restored = jax.tree.unflatten(structure, loaded["flat_params"])
    generated, count = model.generate(restored, ids[:1, :2], max_new_tokens=2, temperature=0.0)
    if count != 2 or generated.shape != (1, 4):
        raise RuntimeError("Checkpoint-backed generation produced unexpected shape")
    receipt = {"scope": "synthetic tokens and randomly initialized tiny CPU model; no math-accuracy claim",
               "steps": int(state.step), "losses": losses, "parameter_delta_l1": delta,
               "checkpoint_parameter_roundtrip": True, "generated_tokens": int(count),
               "generated_shape": list(generated.shape)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/tiny-smoke"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == "__main__":
    main()
