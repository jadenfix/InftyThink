"""Next-token packing must supervise the intended target tokens."""
import numpy as np
import pytest

from src.data.data_converter import _make_instance


def test_loss_mask_aligns_with_shifted_target_tokens():
    instance = _make_instance(
        {"input_ids": np.array([10, 11], dtype=np.int32)},
        {"input_ids": np.array([20, 21], dtype=np.int32)},
        task="segment",
        step=0,
        n_steps=1,
        max_seq_len=6,
        pad_id=0,
    )
    positions = np.flatnonzero(instance["loss_mask"])
    assert positions.tolist() == [1, 2]
    assert instance["target_ids"][positions].tolist() == [20, 21]


def test_full_length_target_does_not_retain_input_or_zero_supervision():
    instance = _make_instance(
        {"input_ids": np.array([10, 11], dtype=np.int32)},
        {"input_ids": np.array([20, 21, 22, 23], dtype=np.int32)},
        task="summary",
        step=0,
        n_steps=1,
        max_seq_len=4,
        pad_id=0,
    )
    assert instance["input_ids"].tolist() == [20, 21, 22, 23]
    positions = np.flatnonzero(instance["loss_mask"])
    assert positions.tolist() == [0, 1, 2]
    assert instance["target_ids"][positions].tolist() == [21, 22, 23]


def test_instance_without_next_token_supervision_is_rejected():
    with pytest.raises(ValueError, match="no supervised next-token target"):
        _make_instance(
            {"input_ids": np.array([], dtype=np.int32)},
            {"input_ids": np.array([20], dtype=np.int32)},
            task="final",
            step=0,
            n_steps=1,
            max_seq_len=1,
            pad_id=0,
        )
