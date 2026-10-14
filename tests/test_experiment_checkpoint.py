"""Scientific evaluation must never silently use random parameters."""
from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

from experiments import _base


def _config(path: Path) -> Path:
    config = {
        "model": {
            "vocab_size": 32,
            "n_layers": 1,
            "d_model": 16,
            "n_heads": 2,
            "d_ff": 32,
            "max_seq_len": 8,
            "dropout": 0.0,
        },
        "data": {"tokenizer": "test-tokenizer"},
    }
    path.write_text(yaml.safe_dump(config))
    return path


def test_missing_checkpoint_fails_before_tokenizer_or_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(_base, "latest_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        _base,
        "load_tokenizer",
        lambda _name: pytest.fail("tokenizer must not load without a checkpoint"),
    )
    with pytest.raises(FileNotFoundError, match="requires a trained checkpoint"):
        _base.load_model_and_params(config_path=str(_config(tmp_path / "config.yaml")))


def test_incompatible_checkpoint_leaf_count_fails_closed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(_base, "load_tokenizer", lambda _name: object())
    import src.model.tokenizer
    monkeypatch.setattr(src.model.tokenizer, "get_vocab_size", lambda _tokenizer: 32)
    monkeypatch.setattr(
        _base,
        "load_checkpoint",
        lambda _path: {"step": 3, "flat_params": [jnp.zeros((1,))]},
    )
    with pytest.raises(ValueError, match="Refusing to evaluate random params"):
        _base.load_model_and_params(
            checkpoint_path=str(checkpoint),
            config_path=str(_config(tmp_path / "config.yaml")),
        )
