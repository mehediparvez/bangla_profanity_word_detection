import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11})


def plot_model_comparison(results, save_path=None):
    set_style()
    models = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall"]

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        vals = [results[model].get(m, 0) for m in metrics]
        errs = [results[model].get(f"{m}_std", 0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=model, yerr=errs, capsize=3)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_speed(results, save_path=None):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, data in results.items():
        ax.scatter(
            data.get("samples_per_second", 0),
            data.get("accuracy", 0),
            s=120,
            label=model,
            zorder=5,
        )
        ax.annotate(
            model,
            (data.get("samples_per_second", 0), data.get("accuracy", 0)),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    if any(d.get("samples_per_second", 0) > 0 for d in results.values()):
        ax.set_xscale("log")
    ax.set_xlabel("Inference Speed (samples/sec, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Inference Speed")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_learning_curves(curves, save_path=None):
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, data in curves.items():
        fractions = [d["fraction"] for d in data]
        f1_scores = [d.get("eval_f1", d.get("f1", 0)) for d in data]
        ax.plot(fractions, f1_scores, "o-", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("F1 Score")
    ax.set_title("Learning Curves")
    ax.set_xticks([0.25, 0.5, 1.0])
    ax.set_xticklabels(["25%", "50%", "100%"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(cm_dict, save_path=None):
    set_style()
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    labels = ["Non-toxic", "Toxic"]
    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_error_distribution(error_counts, save_path=None):
    set_style()
    models = list(error_counts.keys())
    fp_counts = [error_counts[m].get("false_positive", 0) for m in models]
    fn_counts = [error_counts[m].get("false_negative", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fp_counts, width, label="False Positives", color="#e74c3c")
    ax.bar(x + width / 2, fn_counts, width, label="False Negatives", color="#3498db")

    ax.set_ylabel("Count")
    ax.set_title("Error Type Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_all_plots(results, curves, cm_dict, error_counts, plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    saved = []
    # Each plot is skipped when the results needed for it have not been produced yet
    if results:
        plot_model_comparison(results, os.path.join(plots_dir, "model_comparison.png"))
        saved.append("model_comparison.png")
    if any(d.get("samples_per_second", 0) > 0 for d in results.values()):
        plot_accuracy_vs_speed(results, os.path.join(plots_dir, "accuracy_vs_speed.png"))
        saved.append("accuracy_vs_speed.png")
    if curves:
        plot_learning_curves(curves, os.path.join(plots_dir, "learning_curves.png"))
        saved.append("learning_curves.png")
    if cm_dict:
        plot_confusion_matrices(cm_dict, os.path.join(plots_dir, "confusion_matrices.png"))
        saved.append("confusion_matrices.png")
    if error_counts:
        plot_error_distribution(error_counts, os.path.join(plots_dir, "error_distribution.png"))
        saved.append("error_distribution.png")
    print(f"Saved {len(saved)} plots to {plots_dir}/: {', '.join(saved)}")
    return saved
