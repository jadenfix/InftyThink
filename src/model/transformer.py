"""Decoder-only transformer in JAX/Flax with RoPE, RMSNorm, SwiGLU."""
from __future__ import annotations
import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from src.model.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.d_model,))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * scale


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, head_dim: int, dtype=jnp.float32) -> jnp.ndarray:
    """Pre-compute cos/sin tables for RoPE.

    Returns array of shape (seq_len, head_dim) where the first half is cos,
    second half is sin — packed as (seq_len, 2, head_dim//2) for convenience.
    """
    assert head_dim % 2 == 0
    theta = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, theta)  # (seq_len, head_dim//2)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return jnp.array(np.stack([cos, sin], axis=1))  # (seq_len, 2, head_dim//2)


def apply_rope(x: jnp.ndarray, rope_cache: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE to query or key tensors.

    Args:
        x: shape (batch, seq_len, n_heads, head_dim)
        rope_cache: shape (seq_len, 2, head_dim//2)
    """
    seq_len = x.shape[1]
    cos = rope_cache[:seq_len, 0, :]   # (seq_len, head_dim//2)
    sin = rope_cache[:seq_len, 1, :]   # (seq_len, head_dim//2)

    x1 = x[..., :x.shape[-1] // 2]   # (batch, seq, heads, head_dim//2)
    x2 = x[..., x.shape[-1] // 2:]

    # Rotate
    rotated = jnp.concatenate([
        x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :],
        x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :],
    ], axis=-1)
    return rotated


# ---------------------------------------------------------------------------
# Causal multi-head attention with RoPE
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_cache: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        B, T, C = x.shape
        cfg = self.config
        H = cfg.n_heads
        D = cfg.head_dim

        # QKV projections
        q = nn.Dense(C, use_bias=False, name="q_proj")(x)  # (B, T, C)
        k = nn.Dense(C, use_bias=False, name="k_proj")(x)
        v = nn.Dense(C, use_bias=False, name="v_proj")(x)

        # Reshape to (B, T, H, D)
        q = q.reshape(B, T, H, D)
        k = k.reshape(B, T, H, D)
        v = v.reshape(B, T, H, D)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cache)
        k = apply_rope(k, rope_cache)

        # Transpose to (B, H, T, D) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(D)
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # (B, H, T, T)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn = jnp.where(causal_mask[None, None, :, :], attn, -1e9)

        if mask is not None:
            attn = jnp.where(mask[:, None, None, :], attn, -1e9)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=cfg.dropout)(attn, deterministic=not train)

        # Aggregate values
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)  # (B, H, T, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        out = nn.Dense(C, use_bias=False, name="out_proj")(out)
        return out


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        cfg = self.config
        # SwiGLU: gate projection * SiLU(up projection)
        gate = nn.Dense(cfg.d_ff, use_bias=False, name="gate_proj")(x)
        up   = nn.Dense(cfg.d_ff, use_bias=False, name="up_proj")(x)
        x    = jax.nn.silu(gate) * up
        x    = nn.Dropout(rate=cfg.dropout)(x, deterministic=not train)
        x    = nn.Dense(cfg.d_model, use_bias=False, name="down_proj")(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        rope_cache: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        cfg = self.config
        # Pre-norm attention
        residual = x
        x = RMSNorm(cfg.d_model, name="norm1")(x)
        x = MultiHeadAttention(cfg, name="attn")(x, rope_cache, mask=mask, train=train)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=not train)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = RMSNorm(cfg.d_model, name="norm2")(x)
        x = FeedForward(cfg, name="ffn")(x, train=train)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=not train)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Full causal language model
# ---------------------------------------------------------------------------

class CausalLM(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        train: bool = False,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) integer token IDs
            train: whether in training mode (enables dropout)
            mask: optional padding mask (batch, seq_len), True = keep

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        cfg = self.config
        B, T = input_ids.shape

        # Token embedding weight — declared explicitly so we can tie to lm_head
        embed_weight = self.param(
            "token_embed_weight",
            nn.initializers.normal(0.02),
            (cfg.vocab_size, cfg.d_model),
        )
        x = embed_weight[input_ids]  # (B, T, d_model)

        # Pre-compute RoPE cache
        rope_cache = build_rope_cache(T, cfg.head_dim)

        # Transformer layers
        for i in range(cfg.n_layers):
            x = TransformerBlock(cfg, name=f"layer_{i}")(
                x, rope_cache, mask=mask, train=train
            )

        # Final norm
        x = RMSNorm(cfg.d_model, name="final_norm")(x)

        # Output projection (optionally tied to embedding)
        if cfg.tie_embeddings:
            # Tied: reuse embed_weight transposed — no extra parameters
            logits = x @ embed_weight.T
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head")(x)

        return logits  # (B, T, vocab_size)

    def generate(
        self,
        params,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> tuple[jnp.ndarray, int]:
        """Autoregressive generation with nucleus sampling or greedy decoding.

        Args:
            params: model parameters
            input_ids: (1, prompt_len) integer array
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (1.0 = unscaled; 0.0 = greedy)
            top_p: nucleus sampling threshold
            eos_token_id: stop generation on this token
            rng: JAX PRNG key (required for sampling; greedy if None)

        Returns:
            (generated_ids, n_tokens_generated)
            generated_ids: (1, prompt_len + n_tokens_generated)
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        ids = input_ids
        n_generated = 0

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            ids_in = ids[:, -self.config.max_seq_len:]
            logits = self.apply({"params": params}, ids_in)  # (1, T, V)
            next_logits = logits[:, -1, :]                    # (1, V)

            if temperature == 0.0:
                next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
            else:
                next_logits = next_logits / temperature
                rng, sub = jax.random.split(rng)
                next_token = _nucleus_sample(next_logits, top_p, sub)

            ids = jnp.concatenate([ids, next_token], axis=1)
            n_generated += 1

            if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
                break

        return ids, n_generated


# ---------------------------------------------------------------------------
# Nucleus sampling helper
# ---------------------------------------------------------------------------

def _nucleus_sample(
    logits: jnp.ndarray,
    top_p: float,
    rng: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Sample one token using nucleus (top-p) sampling.

    Args:
        logits: (1, vocab_size) float array
        top_p: cumulative probability threshold
        rng: JAX PRNG key

    Returns:
        (1, 1) integer array with the sampled token id
    """
    probs = jax.nn.softmax(logits, axis=-1)  # (1, V)
    sorted_indices = jnp.argsort(probs, axis=-1)[:, ::-1]  # descending
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative = jnp.cumsum(sorted_probs, axis=-1)
    # Mask tokens beyond top_p
    mask = cumulative - sorted_probs < top_p  # keep until cumsum exceeds top_p
    sorted_probs = jnp.where(mask, sorted_probs, 0.0)
    sorted_probs = sorted_probs / jnp.sum(sorted_probs, axis=-1, keepdims=True)

    sampled_sorted_idx = jax.random.choice(
        rng, sorted_probs.shape[-1], p=sorted_probs[0]
    )
    token_id = sorted_indices[0, sampled_sorted_idx]
    return token_id[None, None]  # (1, 1)
