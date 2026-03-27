"""Tokenizer utilities with special InftyThink control tokens."""
from __future__ import annotations
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data.data_converter import CONTROL_TOKENS


def load_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizer:
    """Load a HuggingFace tokenizer and add InftyThink special tokens.

    Adds CONTROL_TOKENS = ["<SEGMENT>", "<SUMMARY>", "<FINAL>"] as
    additional special tokens. Sets pad_token = eos_token.

    Returns:
        tokenizer with updated vocab (vocab_size += len(CONTROL_TOKENS))
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad: bool = True,
    return_attention_mask: bool = True,
) -> dict:
    """Encode text to padded input_ids.

    Returns:
        {"input_ids": np.array (max_length,), "attention_mask": np.array (max_length,)}
    """
    out = tokenizer(
        text,
        max_length=max_length,
        padding="max_length" if pad else False,
        truncation=True,
        return_attention_mask=return_attention_mask,
        return_tensors="np",
    )
    return {
        "input_ids": out["input_ids"][0].astype(np.int32),
        "attention_mask": out["attention_mask"][0].astype(np.int32)
        if return_attention_mask
        else None,
    }


def decode(
    ids: np.ndarray | list[int],
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = True,
) -> str:
    """Decode token ids to a string."""
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def get_vocab_size(tokenizer: PreTrainedTokenizer) -> int:
    """Return the full vocabulary size including added special tokens."""
    return len(tokenizer)
