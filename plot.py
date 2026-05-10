"""
plot.py
Visualization utilities for the seminar project.
- Training/validation loss and accuracy curves
- Confusion matrices
- Metric comparison bar charts (FordA_TEST vs FordB_TEST)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history, model_name, save_dir="plots"):
    """Plot loss and accuracy curves from training history."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "b--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_ylabel("Accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="upper right")

    plt.title(f"{model_name} Training History")
    fig.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_history.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved plot: {path}")


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, save_dir="plots"):
    """Plot and save a confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["-1 (Class 0)", "1 (Class 1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} – {dataset_name}")
    path = os.path.join(save_dir, f"{model_name}_{dataset_name}_cm.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_metric_comparison(metrics_dict, save_dir="plots"):
    """
    Bar chart comparing CNN1D vs LSTM on FordA_TEST and FordB_TEST.
    metrics_dict format:
    {
        "CNN1D": {"FordA_TEST": {...}, "FordB_TEST": {...}},
        "LSTM":  {"FordA_TEST": {...}, "FordB_TEST": {...}}
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    metric_names = ["accuracy", "precision", "recall", "f1"]
    models = ["CNN1D", "LSTM"]
    datasets = ["FordA_TEST", "FordB_TEST"]

    x = np.arange(len(metric_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"CNN1D_FordA_TEST": "#1f77b4", "CNN1D_FordB_TEST": "#aec7e8",
              "LSTM_FordA_TEST": "#ff7f0e", "LSTM_FordB_TEST": "#ffbb78"}

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            vals = [metrics_dict[model][dataset][m] for m in metric_names]
            offset = (i * 2 + j - 1.5) * width
            label = f"{model} ({dataset})"
            ax.bar(x + offset, vals, width, label=label, color=colors[f"{model}_{dataset}"])

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Metrics on FordA_TEST vs FordB_TEST")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def generate_all_plots():
    """Convenience function to generate all plots from saved results."""
    # Load histories
    with open("results/histories.json", "r") as f:
        histories = json.load(f)

    for model_name, hist in histories.items():
        plot_training_history(hist, model_name)

    # Load metrics and re-run confusion matrices
    # (Confusion matrices need raw predictions; here we just plot the metric bars)
    with open("results/metrics.json", "r") as f:
        metrics = json.load(f)

    plot_metric_comparison(metrics)

    print("\nAll plots generated in plots/")


if __name__ == "__main__":
    generate_all_plots()