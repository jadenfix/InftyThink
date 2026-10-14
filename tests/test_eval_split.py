"""Research evaluation must start after the configured training rows."""
import yaml

from experiments import _base


def test_eval_loader_uses_configured_train_boundary(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"data": {"n_train": 5000, "seed": 73}}))
    observed = {}

    def fake_load_openr1(*, n_train, n_eval, seed):
        observed.update(n_train=n_train, n_eval=n_eval, seed=seed)
        return {
            "train": list(range(n_train)),
            "eval": list(range(n_train, n_train + n_eval)),
        }

    import src.data.dataset_loader
    monkeypatch.setattr(src.data.dataset_loader, "load_openr1", fake_load_openr1)
    eval_rows = _base.load_eval_dataset(
        n_eval=500,
        config_path=str(config),
    )
    assert observed == {"n_train": 5000, "n_eval": 500, "seed": 73}
    assert eval_rows[0] == 5000
    assert eval_rows[-1] == 5499
    assert set(range(5000)).isdisjoint(eval_rows)
