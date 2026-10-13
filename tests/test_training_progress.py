"""Training must either make optimizer progress or reject insufficient input."""
import json

import numpy as np
import pytest

from src.model.config import ModelConfig
from src.model.transformer import CausalLM
from src.training.checkpointer import latest_checkpoint, load_checkpoint
from src.training.trainer import TrainConfig, train


def config(tmp_path, **values):
    defaults = dict(batch_size=2, grad_accumulation_steps=2, max_steps=1,
                    warmup_steps=0, eval_every=1, save_every=1,
                    checkpoint_dir=str(tmp_path / "checkpoints"),
                    log_dir=str(tmp_path / "logs" / "training.jsonl"))
    defaults.update(values)
    return TrainConfig(**defaults)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_insufficient_accumulated_batch_rejected_before_initialization(tmp_path, count):
    with pytest.raises(ValueError, match="at least 4 training instances"):
        train(None, None, [{}] * count, [], config(tmp_path))
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("batch_size", -1),
                                          ("grad_accumulation_steps", 0),
                                          ("grad_accumulation_steps", -1)])
def test_invalid_batch_dimensions_rejected(tmp_path, field, value):
    with pytest.raises(ValueError, match="positive"):
        train(None, None, [], [], config(tmp_path, **{field: value}))
    assert not (tmp_path / "checkpoints").exists()


def test_exact_accumulated_batch_runs_training_eval_and_checkpoint(tmp_path):
    model_cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2,
                            d_ff=32, max_seq_len=8, dropout=0.0)
    ids = np.arange(8, dtype=np.int32)
    instances = [dict(input_ids=ids, target_ids=(ids + 1) % 32,
                      loss_mask=np.ones(8, dtype=np.float32), task=task)
                 for task in ("segment", "summary", "final", "segment")]
    state = train(CausalLM(model_cfg), model_cfg, instances, instances, config(tmp_path))
    assert int(state.step) == 1
    records = [json.loads(line) for line in (tmp_path / "logs" / "training.jsonl")
               .read_text().splitlines()]
    assert len(records) == 2  # One actual training update and one evaluation.
    assert all(record["step"] == 1 and np.isfinite(record["loss"]) for record in records)
    assert records[1]["type"] == "eval"
    checkpoint = latest_checkpoint(str(tmp_path / "checkpoints"))
    assert checkpoint is not None
    loaded = load_checkpoint(checkpoint)
    assert loaded["step"] == 1
    assert loaded["n_leaves"] > 0
