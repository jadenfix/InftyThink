"""Checkpoint publication must be complete and non-destructive."""
import json

import jax.numpy as jnp
import pytest

from src.training.checkpointer import latest_checkpoint, list_checkpoints, save_checkpoint


class _State:
    params = {"weight": jnp.arange(4)}


def test_incomplete_or_malformed_checkpoint_is_not_discoverable(tmp_path):
    incomplete = tmp_path / "step_0000009"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text(
        json.dumps({"step": 9, "n_leaves": 1})
    )
    malformed = tmp_path / "step_latest"
    malformed.mkdir()
    (malformed / "metadata.json").write_text("{}")
    (malformed / "params.npz").write_bytes(b"not a checkpoint")
    assert list_checkpoints(str(tmp_path)) == []
    assert latest_checkpoint(str(tmp_path)) is None


def test_corrupt_parameter_archive_is_not_discoverable(tmp_path):
    corrupt = tmp_path / "step_0000010"
    corrupt.mkdir()
    (corrupt / "metadata.json").write_text(
        json.dumps({"step": 10, "n_leaves": 1})
    )
    (corrupt / "params.npz").write_bytes(b"not an npz archive")
    assert list_checkpoints(str(tmp_path)) == []


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    existing = tmp_path / "step_0000003"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_checkpoint(_State(), 3, str(tmp_path))
    assert sentinel.read_text() == "preserve me"


def test_complete_checkpoint_is_published_and_discovered(tmp_path):
    checkpoint = save_checkpoint(_State(), 4, str(tmp_path))
    assert list_checkpoints(str(tmp_path)) == [checkpoint]
    assert latest_checkpoint(str(tmp_path)) == checkpoint
