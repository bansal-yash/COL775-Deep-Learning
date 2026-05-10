import json
import argparse
import matplotlib.pyplot as plt

plot_configs = [
    ("losses", "Loss", False),
    ("bleu", "BLEU Score", True),
    ("chrf", "chrF Score", True),
    ("ter", "TER Score", False),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="metrics_plot.png")
    args = parser.parse_args()

    with open(args.metrics_path) as f:
        metrics = json.load(f)

    exp_name = args.metrics_path.split("metrics_")[-1].replace(".json", "")

    bleu_vals = metrics["bleu"]
    best_ep = bleu_vals.index(max(bleu_vals))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {exp_name}", fontsize=14)

    for ax, (key, label, higher_is_better) in zip(axes.flat, plot_configs):
        if key == "losses":
            train, val = metrics["losses"]
        else:
            train, val = None, metrics[key]

        epochs = range(1, len(val) + 1)

        if train is not None:
            ax.plot(epochs, train, label="Train")
        ax.plot(epochs, val, label="Validation")

        ax.axvline(best_ep + 1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(label)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)
        ax.grid(True)

        if key == "losses":
            t_score = train[best_ep]
            v_score = val[best_ep]
            legend_title = (
                f"@ best BLEU (ep {best_ep + 1})\n"
                f"Train: {t_score:.4f}  Val: {v_score:.4f}"
            )
        else:
            v_score = val[best_ep]
            legend_title = f"@ best BLEU (ep {best_ep + 1})\n" f"Val: {v_score:.4f}"

        ax.legend(title=legend_title, title_fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {args.output_path}")
