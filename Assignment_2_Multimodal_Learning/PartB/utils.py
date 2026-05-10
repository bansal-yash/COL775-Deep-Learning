import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(name):
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(precision):
    enabled = torch.cuda.is_available() and str(precision).lower() in {"fp16", "bf16"}
    dtype = get_dtype(precision)
    return torch.autocast("cuda", dtype=dtype, enabled=enabled)


def make_grad_scaler(precision):
    return torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and str(precision).lower() == "fp16")


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_trainable(module, trainable):
    for p in module.parameters():
        p.requires_grad = trainable


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def exact_match(preds, refs):
    if not refs:
        return 0.0
    return sum(normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs)) / len(refs)


def caption_bleu(preds, refs):
    try:
        import sacrebleu

        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return None


def format_metric(value):
    return "NA" if value is None else f"{value:.4f}"


def save_training_logs(stage, history, out_dir=None):
    log_dir = os.path.join(out_dir, "logs") if out_dir else os.path.join("outputs", "logs")
    save_json(history, os.path.join(log_dir, f"{stage}_logs.json"))


def save_training_plots(stage, history, metric_name, out_dir=None, step_losses=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(os.path.join(out_dir, "plots") if out_dir else os.path.join("outputs", "plots"))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Epoch-level loss curve
    fig, ax = plt.subplots()
    ax.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax.plot(epochs, history["val_loss"], marker="o", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{stage} — Loss per Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{stage}_loss.png"), dpi=150)
    plt.close(fig)

    # Metric curve
    metric_values = history["metric"]
    if any(x is not None for x in metric_values):
        fig, ax = plt.subplots()
        ax.plot(epochs, metric_values, marker="o", color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{stage} — {metric_name} per Epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{stage}_metric.png"), dpi=150)
        plt.close(fig)

    # Per-step loss curve (one entry per optimizer step across all epochs)
    if step_losses:
        fig, ax = plt.subplots()
        ax.plot(step_losses, linewidth=0.8, alpha=0.85)
        ax.set_xlabel("Optimizer Step (global)")
        ax.set_ylabel("Loss")
        ax.set_title(f"{stage} — Loss per Step")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{stage}_step_loss.png"), dpi=150)
        plt.close(fig)


def extract_answer(text):
    text = str(text).strip()
    marker = "answer:"
    low = text.lower()
    if marker in low:
        text = text[low.rfind(marker) + len(marker) :]
    text = text.strip().splitlines()[0] if text.strip() else ""
    return text.strip(" .")


def cosine_warmup_steps(total_steps, warmup_ratio):
    return max(1, int(math.ceil(total_steps * float(warmup_ratio))))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def checkpoint_path(output_dir, name):
    ensure_dir(output_dir)
    return os.path.join(output_dir, name)
