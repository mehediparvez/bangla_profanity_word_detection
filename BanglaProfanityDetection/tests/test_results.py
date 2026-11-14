import os

from src.results import build_evaluation, save_evaluation, collect_results
from src.utils import save_metrics
from src.visualize import generate_all_plots


def test_build_evaluation_counts_errors():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    evaluation = build_evaluation(y_true, y_pred, inference_time=0.5)
    assert evaluation["confusion_matrix"] == [[1, 1], [1, 2]]
    assert evaluation["errors"] == {"false_positive": 1, "false_negative": 1}
    assert evaluation["metrics"]["accuracy"] == 0.6
    assert evaluation["metrics"]["samples_per_second"] == 10.0


def test_build_evaluation_single_class_predictions():
    # A baseline that never fires (e.g. better_profanity on Bangla) still gets a 2x2 matrix
    evaluation = build_evaluation([0, 1, 1], [0, 0, 0])
    assert evaluation["confusion_matrix"] == [[1, 0], [2, 0]]
    assert evaluation["errors"]["false_negative"] == 2
    assert "samples_per_second" not in evaluation["metrics"]


def test_collect_results_and_plots(tmp_path):
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")

    save_evaluation(build_evaluation([0, 1, 1, 0], [0, 0, 0, 0], 0.01), results_dir, "better_profanity")
    save_evaluation(build_evaluation([0, 1, 1, 0], [0, 1, 1, 1], 2.0), results_dir, "banglabert")
    save_metrics(
        {"eval_accuracy": {"mean": 0.8, "std": 0.02}, "eval_f1": {"mean": 0.79, "std": 0.03}},
        os.path.join(results_dir, "banglabert", "summary.json"),
    )
    save_metrics(
        [{"fraction": 0.5, "eval_f1": 0.7}, {"fraction": 1.0, "eval_f1": 0.79}],
        os.path.join(results_dir, "banglabert", "learning_curve.json"),
    )

    comparison, curves, cm_dict, error_counts = collect_results(results_dir)

    assert set(comparison) == {"better_profanity", "BanglaBERT"}
    # Fine-tuned models report the multi-run mean/std, not the single evaluation
    assert comparison["BanglaBERT"]["accuracy"] == 0.8
    assert comparison["BanglaBERT"]["accuracy_std"] == 0.02
    assert comparison["BanglaBERT"]["samples_per_second"] == 2.0
    assert comparison["better_profanity"]["accuracy"] == 0.5
    assert list(curves) == ["BanglaBERT"]
    assert error_counts["better_profanity"] == {"false_positive": 0, "false_negative": 2}

    saved = generate_all_plots(comparison, curves, cm_dict, error_counts, plots_dir)
    assert len(saved) == 5
    for name in saved:
        assert os.path.exists(os.path.join(plots_dir, name))


def test_collect_results_empty(tmp_path):
    comparison, curves, cm_dict, error_counts = collect_results(str(tmp_path))
    assert comparison == {} and curves == {} and cm_dict == {} and error_counts == {}
