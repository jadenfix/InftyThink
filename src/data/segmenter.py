"""Token-based segmentation of reasoning traces."""
from __future__ import annotations
import numpy as np
from transformers import PreTrainedTokenizer


def segment_trace(
    trace: str,
    tokenizer: PreTrainedTokenizer,
    segment_len: int = 128,
    overlap: int = 0,
) -> list[str]:
    """Split a reasoning trace into token-bounded segments.

    Args:
        trace: Raw text of the full reasoning trace.
        tokenizer: HuggingFace tokenizer used to count tokens.
        segment_len: Maximum tokens per segment (≤ segment_len).
        overlap: Number of tokens to repeat from the end of the
                 previous segment at the start of the next (default 0).

    Returns:
        List of string segments. Each segment decodes to ≤ segment_len tokens.
    """
    if overlap >= segment_len:
        raise ValueError(f"overlap ({overlap}) must be < segment_len ({segment_len})")

    token_ids = tokenizer.encode(trace, add_special_tokens=False)
    if not token_ids:
        return []

    segments: list[str] = []
    stride = segment_len - overlap
    start = 0
    while start < len(token_ids):
        end = min(start + segment_len, len(token_ids))
        chunk_ids = token_ids[start:end]
        segments.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if end == len(token_ids):
            break
        start += stride

    return segments


def compute_segment_stats(
    segments: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """Compute token-level statistics over a list of segments.

    Returns:
        {
            "n_segments": int,
            "mean_tokens": float,
            "std_tokens": float,
            "min_tokens": int,
            "max_tokens": int,
            "p25_tokens": float,
            "p50_tokens": float,
            "p75_tokens": float,
        }
    """
    lengths = np.array(
        [len(tokenizer.encode(s, add_special_tokens=False)) for s in segments],
        dtype=np.float32,
    )
    if len(lengths) == 0:
        return {k: 0 for k in ("n_segments", "mean_tokens", "std_tokens",
                                "min_tokens", "max_tokens",
                                "p25_tokens", "p50_tokens", "p75_tokens")}
    return {
        "n_segments": int(len(lengths)),
        "mean_tokens": float(np.mean(lengths)),
        "std_tokens": float(np.std(lengths)),
        "min_tokens": int(np.min(lengths)),
        "max_tokens": int(np.max(lengths)),
        "p25_tokens": float(np.percentile(lengths, 25)),
        "p50_tokens": float(np.percentile(lengths, 50)),
        "p75_tokens": float(np.percentile(lengths, 75)),
    }


def compute_trace_token_length(trace: str, tokenizer: PreTrainedTokenizer) -> int:
    """Return the number of tokens in a full reasoning trace."""
    return len(tokenizer.encode(trace, add_special_tokens=False))
