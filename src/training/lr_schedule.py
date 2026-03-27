"""Learning rate schedules."""
from __future__ import annotations
import optax


def cosine_schedule_with_warmup(
    base_lr: float = 3e-4,
    warmup_steps: int = 200,
    total_steps: int = 10_000,
    min_lr_ratio: float = 0.1,
) -> optax.Schedule:
    """Cosine decay schedule with linear warmup.

    Learning rate ramps linearly from 0 → base_lr over warmup_steps,
    then follows cosine decay from base_lr → min_lr_ratio * base_lr.

    Args:
        base_lr: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps (warmup + decay).
        min_lr_ratio: Fraction of base_lr at the end of training.

    Returns:
        An optax schedule callable: step → learning_rate.
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr_ratio,
    )
    return optax.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps],
    )
