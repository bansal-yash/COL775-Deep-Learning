import random
import numpy as np
import json
import torch
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from model import (
    in100_dataset,
    resnet18,
    no_norm,
    batch_norm,
    batch_instance_norm,
    group_norm,
    layer_norm,
    instance_norm,
)

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)


def compute_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)

    accuracy = 100 * accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    return {
        "accuracies": accuracy,
        "precisions_macro": precision_macro,
        "recalls_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


def plot_metrics(epochs_range, train_vals, val_vals, ylabel, title):
    plt.plot(epochs_range, train_vals, label="Train")
    plt.plot(epochs_range, val_vals, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()


def round_metrics(metrics, decimals=4):
    def round_value(value):
        if isinstance(value, float):
            return round(value, decimals)
        elif isinstance(value, list):
            return [round_value(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(round_value(v) for v in value)
        elif isinstance(value, dict):
            return {k: round_value(v) for k, v in value.items()}
        return value

    return {key: round_value(values) for key, values in metrics.items()}


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    exp,
    save_dir,
):
    metrics = {
        "losses": ([], []),
        "accuracies": ([], []),
        "precisions_macro": ([], []),
        "recalls_macro": ([], []),
        "f1_macro": ([], []),
        "f1_micro": ([], []),
    }

    best_val_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} - Training", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)

            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_metrics = compute_metrics(train_labels, train_preds)

        scheduler.step()

        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            val_preds, val_labels = [], []

            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{epochs} - Validation", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            val_metrics = compute_metrics(val_labels, val_preds)

        running_train_loss /= len(train_loader)
        running_val_loss /= len(val_loader)

        metrics["losses"][0].append(running_train_loss)
        metrics["losses"][1].append(running_val_loss)

        for key in metrics.keys():
            if key != "losses":
                metrics[key][0].append(train_metrics[key])
                metrics[key][1].append(val_metrics[key])

        print(
            f"Epoch [{epoch}/{epochs}]:\n"
            f"Train Loss: {metrics['losses'][0][-1]:.4f}, "
            f"Acc: {metrics['accuracies'][0][-1]:.4f}%, "
            f"F1: {metrics['f1_macro'][0][-1]:.4f}\n"
            f"Val Loss: {metrics['losses'][1][-1]:.4f}, "
            f"Acc: {metrics['accuracies'][1][-1]:.4f}%, "
            f"F1: {metrics['f1_macro'][1][-1]:.4f}\n"
        )

        val_accuracy = metrics["accuracies"][1][-1]

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(
                f"Best Model saving at epoch: {epoch} with Val accuracy: {val_accuracy} %\n"
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
            }

            torch.save(checkpoint, os.path.join(save_dir, f"{exp}.pth"))

        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument(
        "--norm_type",
        type=str,
        required=True,
        choices=["Baseline", "NN", "BN", "IN", "BIN", "LN", "GN"],
    )
    parser.add_argument("--save_dir", type=str, default=".")
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    exp = args.norm_type
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    classes = sorted(os.listdir(train_path))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_batch_size = 128
    val_batch_size = 256

    train_dataset = in100_dataset(train_path, classes, class_to_idx, train_transform)
    val_dataset = in100_dataset(val_path, classes, class_to_idx, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    norm_map = {
        "Baseline": nn.BatchNorm2d,
        "NN": no_norm,
        "BN": batch_norm,
        "IN": instance_norm,
        "BIN": batch_instance_norm,
        "LN": layer_norm,
        "GN": group_norm,
    }

    model = resnet18(num_classes=num_classes, norm_layer=norm_map[exp])
    model = model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-4
    )

    train_metrics = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        exp,
        save_dir,
    )

    with open(os.path.join(save_dir, f"{exp}_metrics.json"), "w") as f:
        json.dump(round_metrics(train_metrics), f, indent=4)

    epochs_range = range(1, num_epochs + 1)

    metric_names = [
        "accuracies",
        "losses",
        "precisions_macro",
        "recalls_macro",
        "f1_macro",
        "f1_micro",
    ]

    metric_labels = [
        "Accuracy (%)",
        "Loss",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1 Score (Macro)",
        "F1 Score (Micro)",
    ]

    plots_dir = os.path.join(save_dir, f"{exp}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for metric, label in zip(metric_names, metric_labels):
        plt.figure(figsize=(8, 5))
        plot_metrics(epochs_range, *train_metrics[metric], label, f"{label} vs Epochs")
        plt.savefig(os.path.join(plots_dir, f"{exp}_{metric}.png"), bbox_inches="tight")
        plt.close()
