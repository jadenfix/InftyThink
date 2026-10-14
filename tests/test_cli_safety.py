"""CLI options must not imply unsupported checkpoint resume."""
from argparse import Namespace

import pytest

import main


def test_training_checkpoint_resume_fails_before_loading_or_writing():
    with pytest.raises(ValueError, match="optimizer state"):
        main.cmd_train(Namespace(checkpoint="checkpoints/step_0000001"))


def test_training_eval_cap_retains_final_task():
    instances = [
        {"task": "segment"},
        {"task": "summary"},
        {"task": "segment"},
        {"task": "summary"},
        {"task": "segment"},
        {"task": "summary"},
        {"task": "final"},
    ]
    selected = main._select_training_eval_instances(instances)
    assert len(selected) == 4
    assert [item["task"] for item in selected] == [
        "segment", "summary", "segment", "final"
    ]
