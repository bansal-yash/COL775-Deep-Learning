import os
import csv
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--predictions_csv", type=str, required=True)
    args = parser.parse_args()

    gt = {}
    sorted_classes = sorted(
        [
            cls_name
            for cls_name in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, cls_name))
        ]
    )
    class_to_sorted_idx = {cls: str(i) for i, cls in enumerate(sorted_classes)}

    for cls_name in sorted_classes:
        cls_folder = os.path.join(args.data_path, cls_name)
        for fname in os.listdir(cls_folder):
            image_name = os.path.join(cls_name, fname)
            gt[image_name] = class_to_sorted_idx[cls_name]

    true_labels, pred_labels = [], []
    with open(args.predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_name"]
            pred_class = row["predicted_label"]
            if image_name not in gt:
                print(f"Warning: {image_name} not found in data_path, skipping.")
                continue
            true_labels.append(gt[image_name])
            pred_labels.append(pred_class)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = 100 * accuracy_score(true_labels, pred_labels)
    precision_macro = precision_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_micro = f1_score(true_labels, pred_labels, average="micro", zero_division=0)

    print(f"Accuracy         : {accuracy:.4f}%")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro)   : {recall_macro:.4f}")
    print(f"F1 (Macro)       : {f1_macro:.4f}")
    print(f"F1 (Micro)       : {f1_micro:.4f}")
