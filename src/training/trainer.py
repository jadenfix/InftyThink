"""JAX training loop with JIT-compiled train_step and gradient accumulation."""
from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

from src.model.config import ModelConfig
from src.model.transformer import CausalLM
from src.training.losses import segment_summary_loss, cross_entropy_loss
from src.training.lr_schedule import cosine_schedule_with_warmup
from src.training.checkpointer import save_checkpoint, latest_checkpoint


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accumulation_steps: int = 4    # effective batch = batch_size * grad_accumulation_steps
    max_steps: int = 10_000
    eval_every: int = 500
    save_every: int = 1_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    seed: int = 42
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "results/training_logs.jsonl"


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class InftyThinkTrainState(train_state.TrainState):
    pass  # use standard Flax TrainState; extend if needed


def create_train_state(
    model: CausalLM,
    config: TrainConfig,
    model_config: ModelConfig,
    rng: jax.random.PRNGKey,
) -> InftyThinkTrainState:
    """Initialize model params and optimizer state."""
    dummy_input = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    schedule = cosine_schedule_with_warmup(
        base_lr=config.learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            b1=0.9,
            b2=0.95,
        ),
    )

    return InftyThinkTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


# ---------------------------------------------------------------------------
# Single micro-batch gradient computation (JIT-compiled, no update)
# ---------------------------------------------------------------------------

@jax.jit
def _compute_grads(
    state: InftyThinkTrainState,
    batch: dict,
    dropout_rng: jax.random.PRNGKey,
) -> tuple[Any, jnp.ndarray, dict]:
    """Compute gradients for one micro-batch without applying them.

    Returns:
        (grads pytree, loss scalar, breakdown dict)
    """
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            train=True,
            rngs={"dropout": dropout_rng},
        )
        total_loss, breakdown = segment_summary_loss(
            logits,
            batch["target_ids"],
            batch["loss_mask"],
            batch["task_flags"],
        )
        return total_loss, breakdown

    (loss, breakdown), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return grads, loss, breakdown


@jax.jit
def _apply_grads(
    state: InftyThinkTrainState,
    grads,
) -> InftyThinkTrainState:
    """Apply accumulated gradients to the optimizer state."""
    return state.apply_gradients(grads=grads)


def train_step(
    state: InftyThinkTrainState,
    micro_batches: list[dict],
    dropout_rng: jax.random.PRNGKey,
) -> tuple[InftyThinkTrainState, dict]:
    """One optimizer step over `grad_accumulation_steps` micro-batches.

    Gradients are accumulated (averaged) across all micro-batches before
    the parameter update, matching an effective batch size of
    batch_size * grad_accumulation_steps.

    Args:
        state: current TrainState
        micro_batches: list of micro-batch dicts, each with:
            "input_ids":  (micro_batch, seq_len) int32
            "target_ids": (micro_batch, seq_len) int32
            "loss_mask":  (micro_batch, seq_len) float32
            "task_flags": (micro_batch,) int32
        dropout_rng: JAX PRNG key (split per micro-batch)

    Returns:
        (updated_state, metrics_dict)
    """
    n = len(micro_batches)
    accumulated_grads = None
    total_loss = 0.0
    total_seg_loss = 0.0
    total_sum_loss = 0.0
    total_fin_loss = 0.0

    for i, mb in enumerate(micro_batches):
        rng_i = jax.random.fold_in(dropout_rng, i)
        grads, loss, breakdown = _compute_grads(state, mb, rng_i)

        total_loss += float(loss) / n
        total_seg_loss += breakdown["segment_loss"] / n
        total_sum_loss += breakdown["summary_loss"] / n
        total_fin_loss += breakdown["final_loss"] / n

        if accumulated_grads is None:
            accumulated_grads = jax.tree_util.tree_map(lambda g: g / n, grads)
        else:
            accumulated_grads = jax.tree_util.tree_map(
                lambda acc, g: acc + g / n, accumulated_grads, grads
            )

    state = _apply_grads(state, accumulated_grads)

    metrics = {
        "loss": float(total_loss),
        "segment_loss": float(total_seg_loss),
        "summary_loss": float(total_sum_loss),
        "final_loss": float(total_fin_loss),
        "step": int(state.step),
    }
    return state, metrics


# ---------------------------------------------------------------------------
# Evaluation step (no grad)
# ---------------------------------------------------------------------------

@jax.jit
def eval_step(
    state: InftyThinkTrainState,
    batch: dict,
) -> dict:
    """Evaluate a batch without gradient updates."""
    logits = state.apply_fn({"params": state.params}, batch["input_ids"], train=False)
    total_loss, breakdown = segment_summary_loss(
        logits,
        batch["target_ids"],
        batch["loss_mask"],
        batch["task_flags"],
    )
    return {
        "loss": total_loss,
        "segment_loss": jnp.array(breakdown["segment_loss"]),
        "summary_loss": jnp.array(breakdown["summary_loss"]),
        "final_loss": jnp.array(breakdown["final_loss"]),
    }


# ---------------------------------------------------------------------------
# DataLoader-style batch iterator
# ---------------------------------------------------------------------------

def make_batches(
    instances: list[dict],
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> list[dict]:
    """Shuffle instances and yield batches as numpy dicts."""
    if shuffle:
        indices = rng.permutation(len(instances))
    else:
        indices = np.arange(len(instances))

    batches = []
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        idx = indices[start: start + batch_size]
        batch = {
            "input_ids":  np.stack([instances[i]["input_ids"]  for i in idx]),
            "target_ids": np.stack([instances[i]["target_ids"] for i in idx]),
            "loss_mask":  np.stack([instances[i]["loss_mask"]  for i in idx]),
            "task_flags": np.array(
                [{"segment": 0, "summary": 1, "final": 2}[instances[i]["task"]] for i in idx],
                dtype=np.int32,
            ),
        }
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: CausalLM,
    model_config: ModelConfig,
    train_instances: list[dict],
    eval_instances: list[dict],
    config: TrainConfig,
) -> InftyThinkTrainState:
    """Full supervised training loop.

    Args:
        model: CausalLM instance
        model_config: ModelConfig
        train_instances: list of dicts from data_converter.convert_example
        eval_instances: list of dicts for evaluation
        config: TrainConfig

    Returns:
        Final TrainState.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.log_dir) or ".", exist_ok=True)
    log_file = open(config.log_dir, "a")

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, config, model_config, init_rng)

    np_rng = np.random.default_rng(config.seed)
    step = 0
    epoch = 0

    print(f"Starting training: {config.max_steps} steps, "
          f"batch={config.batch_size}, eff_batch={config.batch_size * config.grad_accumulation_steps}")
    print(f"Train instances: {len(train_instances)}, Eval instances: {len(eval_instances)}")

    pbar = tqdm(total=config.max_steps, desc="Training")

    acc_steps = config.grad_accumulation_steps

    while step < config.max_steps:
        epoch += 1
        # Each "batch" is one micro-batch of size batch_size.
        # We group acc_steps micro-batches into one optimizer step.
        micro_batches = make_batches(train_instances, config.batch_size, np_rng, shuffle=True)

        # Iterate in chunks of acc_steps
        for chunk_start in range(0, len(micro_batches) - acc_steps + 1, acc_steps):
            if step >= config.max_steps:
                break

            chunk = micro_batches[chunk_start: chunk_start + acc_steps]
            chunk_jax = [{k: jnp.array(v) for k, v in mb.items()} for mb in chunk]

            rng, dropout_rng = jax.random.split(rng)
            state, metrics = train_step(state, chunk_jax, dropout_rng)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            # Log
            metrics["epoch"] = epoch
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            # Evaluate
            if step % config.eval_every == 0:
                eval_metrics = _run_eval(state, eval_instances, config.batch_size)
                eval_metrics["step"] = step
                eval_metrics["type"] = "eval"
                log_file.write(json.dumps(eval_metrics) + "\n")
                log_file.flush()
                tqdm.write(
                    f"[Step {step}] eval_loss={eval_metrics['loss']:.4f} "
                    f"seg={eval_metrics['segment_loss']:.4f} "
                    f"sum={eval_metrics['summary_loss']:.4f}"
                )

            # Save checkpoint
            if step % config.save_every == 0:
                save_checkpoint(state, step, config.checkpoint_dir)

    pbar.close()
    log_file.close()

    # Final checkpoint
    save_checkpoint(state, step, config.checkpoint_dir)
    print(f"Training complete. Final step: {step}")
    return state


def _run_eval(
    state: InftyThinkTrainState,
    eval_instances: list[dict],
    batch_size: int,
) -> dict:
    """Run evaluation on all eval instances and return mean metrics."""
    np_rng = np.random.default_rng(0)
    batches = make_batches(eval_instances, batch_size, np_rng, shuffle=False)
    if not batches:
        return {"loss": 0.0, "segment_loss": 0.0, "summary_loss": 0.0, "final_loss": 0.0}

    all_metrics: list[dict] = []
    for batch in batches:
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        m = eval_step(state, batch_jax)
        all_metrics.append(m)

    return {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }
