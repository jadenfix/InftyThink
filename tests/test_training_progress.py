"""Training must either make optimizer progress or reject insufficient input."""
import json

import numpy as np
import pytest

from src.model.config import ModelConfig
from src.model.transformer import CausalLM
from src.training.checkpointer import latest_checkpoint, load_checkpoint
from src.training import trainer
from src.training.trainer import TrainConfig, make_batches, train


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
    assert records[1]["examples_evaluated"] == 4
    assert records[1]["batches_evaluated"] == 2
    checkpoint = latest_checkpoint(str(tmp_path / "checkpoints"))
    assert checkpoint is not None
    loaded = load_checkpoint(checkpoint)
    assert loaded["step"] == 1
    assert loaded["n_leaves"] > 0


def _instances(count):
    return [
        {
            "input_ids": np.array([i], dtype=np.int32),
            "target_ids": np.array([i], dtype=np.int32),
            "loss_mask": np.ones(1, dtype=np.float32),
            "task": "segment",
        }
        for i in range(count)
    ]


@pytest.mark.parametrize("count", [1, 7, 9, 500])
def test_evaluation_batches_include_every_tail_example(count):
    batches = make_batches(
        _instances(count),
        batch_size=8,
        rng=np.random.default_rng(0),
        shuffle=False,
        drop_remainder=False,
    )
    observed = [int(value) for batch in batches for value in batch["input_ids"][:, 0]]
    assert observed == list(range(count))


def test_training_batches_still_drop_remainder():
    batches = make_batches(
        _instances(9),
        batch_size=8,
        rng=np.random.default_rng(0),
        shuffle=False,
    )
    assert sum(len(batch["input_ids"]) for batch in batches) == 8


def test_run_eval_counts_tail_and_weights_loss_by_tokens(monkeypatch):
    def fake_eval_step(_state, batch):
        values = np.asarray(batch["input_ids"][:, 0], dtype=np.float32)
        token_count = float(len(values))
        return {
            "loss": float(values.mean()),
            "segment_loss": float(values.mean()),
            "summary_loss": 0.0,
            "final_loss": 0.0,
            "segment_token_count": token_count,
            "summary_token_count": 0.0,
            "final_token_count": 0.0,
            "example_count": len(values),
        }

    monkeypatch.setattr(trainer, "eval_step", fake_eval_step)
    metrics = trainer._run_eval(None, _instances(9), batch_size=8)
    assert metrics["examples_evaluated"] == 9
    assert metrics["batches_evaluated"] == 2
    assert metrics["loss"] == pytest.approx(4.0)
    assert metrics["segment_loss"] == pytest.approx(4.0)
    assert metrics["segment_tokens"] == 9
    assert metrics["summary_tokens"] == 0
    assert metrics["final_tokens"] == 0


def test_run_eval_token_weights_within_tasks_then_equal_weights_tasks(monkeypatch):
    calls = iter([
        {
            "loss": 0.0,
            "segment_loss": 2.0,
            "summary_loss": 10.0,
            "final_loss": 0.0,
            "segment_token_count": 1.0,
            "summary_token_count": 3.0,
            "final_token_count": 0.0,
            "example_count": 2,
        },
        {
            "loss": 0.0,
            "segment_loss": 8.0,
            "summary_loss": 0.0,
            "final_loss": 100.0,
            "segment_token_count": 2.0,
            "summary_token_count": 0.0,
            "final_token_count": 4.0,
            "example_count": 1,
        },
    ])
    monkeypatch.setattr(trainer, "eval_step", lambda _state, _batch: next(calls))
    metrics = trainer._run_eval(None, _instances(3), batch_size=2)
    assert metrics["segment_loss"] == pytest.approx(6.0)
    assert metrics["summary_loss"] == pytest.approx(10.0)
    assert metrics["final_loss"] == pytest.approx(100.0)
    assert metrics["loss"] == pytest.approx((6.0 + 10.0 + 100.0) / 3)
    assert metrics["segment_tokens"] == 3
    assert metrics["summary_tokens"] == 3
    assert metrics["final_tokens"] == 4


def test_empty_evaluation_is_rejected():
    with pytest.raises(ValueError, match="at least one instance"):
        trainer._run_eval(None, [], batch_size=8)


def test_preexisting_final_checkpoint_is_not_silently_accepted(tmp_path):
    model_cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2,
                            d_ff=32, max_seq_len=8, dropout=0.0)
    ids = np.arange(8, dtype=np.int32)
    instances = [dict(input_ids=ids, target_ids=(ids + 1) % 32,
                      loss_mask=np.ones(8, dtype=np.float32), task=task)
                 for task in ("segment", "summary", "final", "segment")]
    checkpoint = tmp_path / "checkpoints" / "step_0000001"
    checkpoint.mkdir(parents=True)
    sentinel = checkpoint / "sentinel.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train(
            CausalLM(model_cfg),
            model_cfg,
            instances,
            instances,
            config(tmp_path, save_every=2),
        )
    assert sentinel.read_text() == "preserve"
