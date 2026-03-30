"""Loss functions for segment and summary training objectives."""
from __future__ import annotations
import jax
import jax.numpy as jnp


def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute masked cross-entropy loss.

    Args:
        logits: (batch, seq_len, vocab_size) float32
        targets: (batch, seq_len) int32  — next-token targets
        mask: (batch, seq_len) float32  — 1.0 on positions to include in loss,
              0.0 on positions to ignore (input tokens, padding).
              If None, all positions are included.

    Returns:
        Scalar mean loss over unmasked positions.
    """
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = log_probs[jnp.arange(len(targets_flat)), targets_flat]

    if mask is not None:
        mask_flat = mask.reshape(-1)
        loss = -jnp.sum(target_log_probs * mask_flat) / (jnp.sum(mask_flat) + 1e-8)
    else:
        loss = -jnp.mean(target_log_probs)

    return loss


def segment_summary_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_mask: jnp.ndarray,
    task_flags: jnp.ndarray,
) -> tuple[jnp.ndarray, dict]:
    """Compute separate segment and summary losses from a mixed batch.

    Args:
        logits:     (batch, seq_len, vocab_size)
        targets:    (batch, seq_len) int32
        loss_mask:  (batch, seq_len) float32 — 1.0 on target token positions
        task_flags: (batch,) int32 — 0=segment, 1=summary, 2=final

    Returns:
        (total_loss, {"segment_loss": float, "summary_loss": float, "final_loss": float})
    """
    seg_mask_batch  = (task_flags == 0).astype(jnp.float32)  # (batch,)
    sum_mask_batch  = (task_flags == 1).astype(jnp.float32)
    fin_mask_batch  = (task_flags == 2).astype(jnp.float32)

    # Combine batch-level task mask with token-level loss mask
    seg_loss_mask = loss_mask * seg_mask_batch[:, None]
    sum_loss_mask = loss_mask * sum_mask_batch[:, None]
    fin_loss_mask = loss_mask * fin_mask_batch[:, None]

    seg_loss = cross_entropy_loss(logits, targets, seg_loss_mask)
    sum_loss = cross_entropy_loss(logits, targets, sum_loss_mask)
    fin_loss = cross_entropy_loss(logits, targets, fin_loss_mask)

    # Total: average over the task types that are actually present in this batch.
    # has_seg/sum/fin is 1.0 if any example of that type appears, else 0.0.
    has_seg = jnp.minimum(jnp.sum(seg_mask_batch), 1.0)
    has_sum = jnp.minimum(jnp.sum(sum_mask_batch), 1.0)
    has_fin = jnp.minimum(jnp.sum(fin_mask_batch), 1.0)
    n_active = has_seg + has_sum + has_fin  # ∈ {1, 2, 3}
    total_loss = (
        seg_loss * has_seg + sum_loss * has_sum + fin_loss * has_fin
    ) / (n_active + 1e-8)

    return total_loss, {
        "segment_loss": seg_loss,
        "summary_loss": sum_loss,
        "final_loss": fin_loss,
    }
