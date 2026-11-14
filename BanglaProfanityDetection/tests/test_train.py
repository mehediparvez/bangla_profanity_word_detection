import json

import src.train as train_mod
from src.train import split_validation, train_model, train_with_multiple_runs


TEXTS = [f"comment {i}" for i in range(40)]
LABELS = [i % 2 for i in range(40)]


def test_split_validation_is_stratified_and_deterministic():
    fit_t, val_t, fit_l, val_l = split_validation(TEXTS, LABELS, val_size=0.1, seed=42)
    assert len(val_t) == 4 and len(fit_t) == 36
    assert sum(val_l) == 2  # both classes represented equally
    assert not set(val_t) & set(fit_t)
    assert split_validation(TEXTS, LABELS, val_size=0.1, seed=42)[1] == val_t


def test_multiple_runs_use_different_seeds(monkeypatch):
    calls = []

    def fake_train_model(*args, **kwargs):
        calls.append(kwargs)
        return {"eval_accuracy": 0.5, "eval_f1": 0.5}, "unused"

    monkeypatch.setattr(train_mod, "train_model", fake_train_model)
    train_with_multiple_runs("org/model", TEXTS, TEXTS, LABELS, LABELS, {}, num_runs=3, seed=42)

    assert [c["seed"] for c in calls] == [43, 44, 45]
    # The validation split stays fixed across runs; only the training seed changes
    assert {c["split_seed"] for c in calls} == {42}
    assert [c["keep_model"] for c in calls] == [True, False, False]


def test_train_model_skips_finished_run(tmp_path):
    run_dir = tmp_path / "run_1"
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "best_model" / "config.json").write_text("{}")
    (run_dir / "best_model" / "model.safetensors").write_bytes(b"")
    (run_dir / "metrics.json").write_text(json.dumps({"eval_f1": 0.9}))

    # Model name is never loaded because the finished run is reused
    metrics, model_dir = train_model(
        "does/not-exist", TEXTS, TEXTS, LABELS, LABELS, {}, output_dir=str(tmp_path)
    )
    assert metrics == {"eval_f1": 0.9}
    assert model_dir.endswith("best_model")


def test_has_saved_model_needs_config_and_weights(tmp_path):
    # A run whose model files were lost (e.g. in a Drive sync) must not count as finished
    assert not train_mod._has_saved_model(str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    assert not train_mod._has_saved_model(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert train_mod._has_saved_model(str(tmp_path))
