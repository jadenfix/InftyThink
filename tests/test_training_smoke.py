"""Real tiny JAX optimizer and checkpoint generation regression."""
from src.training.smoke import run_smoke


def test_tiny_training_checkpoint_and_generation(tmp_path):
    receipt = run_smoke(tmp_path)
    assert receipt["steps"] == 2
    assert receipt["parameter_delta_l1"] > 0
    assert receipt["checkpoint_parameter_roundtrip"]
    assert receipt["generated_tokens"] == 2
    assert receipt["evaluation_examples"] == 9
    assert receipt["evaluation_batches"] == 2
    assert receipt["evaluation_tail_included"]
    assert (tmp_path / "smoke.json").is_file()
