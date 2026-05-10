import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("plots", exist_ok=True)

def plot_hist(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    e = range(1, len(hist["tl"]) + 1)
    ax[0].plot(e, hist["tl"], label="train", lw=1.5)
    ax[0].plot(e, hist["vl"], label="val", lw=1.5)
    ax[0].set_title(f"{name} – Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(e, hist["va"], color="green", lw=1.5)
    ax[1].set_title(f"{name} – Val Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].set_ylim(0, 1.05)
    fig.tight_layout(pad=2.5)
    plt.savefig(f"plots/{name}_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_cm(y, p, name, ds):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, p), display_labels=["-1","1"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} – {ds}")
    plt.tight_layout(pad=2.5)
    plt.savefig(f"plots/{name}_{ds}_cm.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_cmp(res):
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = ["acc", "prec", "rec", "f1"]
    x = np.arange(len(keys)); w = 0.2
    colors = {
        "CNN1D_A": "#1f77b4", "CNN1D_B": "#aec7e8",
        "GRU_A": "#ff7f0e", "GRU_B": "#ffbb78",
    }
    for i, (model, d) in enumerate(res.items()):
        for j, ds in enumerate(["FordA_TEST", "FordB_TEST"]):
            v = [d[ds][k] for k in keys]
            ax.bar(x + (i * 2 + j - 1.5) * w, v, w, label=f"{model} ({ds})", color=colors[f"{model}_{ds[4]}"])
    ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    ax.set_xticks(x); ax.set_xticklabels([k.capitalize() for k in keys])
    ax.set_ylim(0, 1.05); ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout(pad=2.5)
    plt.savefig("plots/comparison.png", dpi=200, bbox_inches="tight")
    plt.close()