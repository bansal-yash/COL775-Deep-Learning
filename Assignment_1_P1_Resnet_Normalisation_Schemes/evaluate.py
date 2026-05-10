import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import (
    resnet18,
    no_norm,
    batch_norm,
    batch_instance_norm,
    group_norm,
    layer_norm,
    instance_norm,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)


class TestDataset(Dataset):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.samples = []

        entries = os.listdir(data_path)
        has_subdirs = any(os.path.isdir(os.path.join(data_path, e)) for e in entries)

        img_exts = {".jpg", ".jpeg"}

        if has_subdirs:
            for folder in sorted(entries):
                folder_path = os.path.join(data_path, folder)
                if not os.path.isdir(folder_path):
                    continue
                for fname in sorted(os.listdir(folder_path)):
                    if os.path.splitext(fname)[1].lower() in img_exts:
                        self.samples.append(
                            (
                                os.path.join(folder, fname),
                                os.path.join(folder_path, fname),
                            )
                        )
        else:
            for fname in sorted(entries):
                if os.path.splitext(fname)[1].lower() in img_exts:
                    self.samples.append(
                        (
                            fname,
                            os.path.join(data_path, fname),
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--norm_type",
        type=str,
        required=True,
        choices=["Baseline", "NN", "BN", "IN", "BIN", "LN", "GN"],
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    norm_map = {
        "Baseline": nn.BatchNorm2d,
        "NN": no_norm,
        "BN": batch_norm,
        "IN": instance_norm,
        "BIN": batch_instance_norm,
        "LN": layer_norm,
        "GN": group_norm,
    }

    model = resnet18(num_classes=num_classes, norm_layer=norm_map[args.norm_type])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    gt = {}
    for cls_name in os.listdir(args.data_path):
        cls_folder = os.path.join(args.data_path, cls_name)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            image_name = os.path.join(cls_name, fname)
            gt[image_name] = cls_name

    dataset = TestDataset(args.data_path, transform)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            for name, pred in zip(image_names, preds):
                pred_class = idx_to_class[pred]
                if name not in gt:
                    print(f"Warning: {name} not found in data_path, skipping.")
                    continue
                true_labels.append(gt[name])
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
