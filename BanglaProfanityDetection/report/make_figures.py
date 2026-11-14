"""Draw the report's result charts as PNGs in report/figures/.

The numbers are the final test results from the notebooks. Update them here and re-run
(python report/make_figures.py) when results change.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# Values from results/<model>/{summary,evaluation,learning_curve}.json on Google Drive.
# name: (F1 mean, F1 std, accuracy of the evaluated model, comments per second)
RESULTS = {
    "better_profanity": (0.2989, 0.0, 0.4684, 21.4),
    "LLM zero-shot": (0.7681, 0.0, 0.7748, 2.6),
    "BanglaBERT": (0.9276, 0.0035, 0.9271, 183.3),  # accuracy and speed: run 1 model
    "XLM-RoBERTa": (0.9430, 0.0013, 0.9413, 177.2),
}
COLORS = ["#9e9e9e", "#e69f00", "#56b4e9", "#0072b2"]

# Confusion matrices [[TN, FP], [FN, TP]] on the test set
CONFUSION = {
    "better_profanity": [[1491, 0], [1692, 0]],
    "LLM zero-shot": [[1391, 29], [654, 959]],
    "BanglaBERT": [[1341, 150], [82, 1610]],
    "XLM-RoBERTa": [[1390, 101], [86, 1606]],
}

# Test F1 when trained on 25 / 50 / 100 % of the training set
LEARNING_CURVE = {
    "BanglaBERT": [0.9061, 0.9158, 0.9274],
    "XLM-RoBERTa": [0.9142, 0.9362, 0.9440],
}


def f1_bar():
    names = list(RESULTS)
    f1 = [RESULTS[n][0] for n in names]
    std = [RESULTS[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(names, f1, xerr=std, color=COLORS, capsize=4)
    for bar, v in zip(bars, f1):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Weighted F1-score")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f1_comparison.png"), dpi=200)
    plt.close(fig)


def training_dynamics():
    # BanglaBERT run 1, per-epoch values from the Trainer log
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.2645, 0.2034, 0.0990, 0.0612, 0.0256]
    val_loss = [0.2595, 0.2784, 0.4122, 0.4665, 0.5774]
    val_f1 = [0.9026, 0.9064, 0.9143, 0.9143, 0.9144]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(epochs, train_loss, "o-", label="Training")
    a.plot(epochs, val_loss, "s-", color="#d62728", label="Validation")
    a.set(title="Loss", xlabel="Epoch", ylabel="Loss", xticks=epochs, ylim=(0, 0.65))
    a.legend()
    b.plot(epochs, val_f1, "^-", color="#2ca02c")
    b.set(title="Validation F1", xlabel="Epoch", ylabel="Weighted F1", xticks=epochs, ylim=(0.89, 0.92))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_dynamics.png"), dpi=200)
    plt.close(fig)


def accuracy_vs_speed():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, (_, _, acc, speed)), color in zip(RESULTS.items(), COLORS):
        ax.scatter(speed, acc, s=110, color=color, edgecolor="black", zorder=3)
        # The two transformers sit close together, so put one label above and one below
        offset = {"XLM-RoBERTa": (8, 6), "BanglaBERT": (8, -14)}.get(name, (8, -4))
        ax.annotate(name, (speed, acc), xytext=offset, textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlim=(1, 800), ylim=(0.4, 1.0), xlabel="Inference speed (comments per second, log scale)", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "accuracy_vs_speed.png"), dpi=200)
    plt.close(fig)


def learning_curves():
    fractions = ["25%\n(3,182)", "50%\n(6,365)", "100%\n(12,730)"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, f1), color, marker in zip(LEARNING_CURVE.items(), COLORS[2:], ["o", "s"]):
        ax.plot(fractions, f1, marker=marker, color=color, linewidth=2, label=name)
        for x, v in zip(fractions, f1):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlabel="Share of training data (comments)", ylabel="Test weighted F1", ylim=(0.89, 0.955))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=200)
    plt.close(fig)


def confusion_matrices():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    labels = ["Non-toxic", "Toxic"]
    for ax, (name, cm) in zip(axes, CONFUSION.items()):
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        top = max(max(row) for row in cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                        color="white" if cm[i][j] > top / 2 else "black", fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels if ax is axes[0] else ["", ""])
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    f1_bar()
    training_dynamics()
    accuracy_vs_speed()
    learning_curves()
    confusion_matrices()
    print("Figures written to", OUT)
