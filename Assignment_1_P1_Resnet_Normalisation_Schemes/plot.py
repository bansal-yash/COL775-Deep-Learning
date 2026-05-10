import json
import argparse
import matplotlib.pyplot as plt

metrics_config = [
    ("losses", "Loss", False),
    ("accuracies", "Accuracy (%)", True),
    ("precisions_macro", "Precision (Macro)", True),
    ("recalls_macro", "Recall (Macro)", True),
    ("f1_macro", "F1 Score (Macro)", True),
    ("f1_micro", "F1 Score (Micro)", True),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", type=str)
    parser.add_argument("--out", type=str, default="metrics.png")
    args = parser.parse_args()

    with open(args.metrics_json) as f:
        data = json.load(f)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for ax, (key, label, show_max) in zip(axes.flat, metrics_config):
        train, val = data[key]
        epochs = range(1, len(train) + 1)
        ax.plot(epochs, train, label="Train")
        ax.plot(epochs, val, label="Validation")
        ax.set_title(label)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)
        ax.grid(True)

        if show_max:
            max_train, max_val = max(train), max(val)
            legend_title = f"Max Train: {max_train:.4f}\nMax Val:   {max_val:.4f}"
            ax.legend(title=legend_title, title_fontsize=8)
        else:
            ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved to {args.out}")
