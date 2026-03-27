"""Model configuration dataclass."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os


@dataclass
class ModelConfig:
    """Configuration for the small decoder-only transformer.

    Parameter budget (approximate):
      Embedding:        vocab_size * d_model       = 32000 * 512  = 16.4M
      Each layer:       12 * d_model^2             = 12 * 262144  = 3.15M
      All layers:       n_layers * 3.15M           = 6 * 3.15M    = 18.9M
      Total:                                                       ≈ 35.3M (tied embeddings)
                                                                   ≈ 51.7M (untied)
    """
    vocab_size: int = 32_000
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8             # head_dim = d_model / n_heads = 64
    d_ff: int = 2048             # feed-forward hidden dim = 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1
    tie_embeddings: bool = True  # share input and output embedding weights

    # Derived (not set by user)
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def param_count_estimate(self) -> int:
        """Return an approximate total parameter count."""
        embed = self.vocab_size * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model   # Q, K, V, O projections
            + 2 * self.d_model * self.d_ff     # FFN up + down (SwiGLU uses 3 matrices)
            + self.d_ff * self.d_model         # SwiGLU gate
            + 2 * self.d_model                 # 2x RMSNorm scale params
        )
        total = embed + self.n_layers * per_layer
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total
