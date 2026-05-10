import random
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import in100_dataset, resnet18, batch_instance_norm

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


def select_best_worst_classes(model, val_loader, class_to_idx):
    num_classes = len(class_to_idx)
    all_true, all_pred, all_conf, all_paths = [], [], [], []

    model.eval()
    with torch.no_grad():
        idx_ptr = 0
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            logits = model(images)

            probs = nn.functional.softmax(logits, dim=1)
            pred_conf, pred_cls = probs.max(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(pred_cls.cpu().numpy())
            all_conf.extend(pred_conf.cpu().numpy())

            batch_size = images.size(0)
            all_paths.extend(
                val_loader.dataset.image_paths[idx_ptr : idx_ptr + batch_size]
            )
            idx_ptr += batch_size

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_conf = np.array(all_conf)
    all_paths = np.array(all_paths)

    per_class_acc = {}
    for cls_idx in range(num_classes):
        mask = all_true == cls_idx
        if mask.sum() == 0:
            per_class_acc[cls_idx] = 0.0
        else:
            per_class_acc[cls_idx] = (all_pred[mask] == cls_idx).mean()

    sorted_by_acc = sorted(per_class_acc.items(), key=lambda x: x[1])
    worst_3 = []
    for cls_idx, _ in sorted_by_acc:
        mask = all_true == cls_idx
        correct = (mask & (all_pred == cls_idx)).sum()

        if correct >= 5:
            worst_3.append(cls_idx)

        if len(worst_3) == 3:
            break

    sorted_by_acc_desc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

    best_3 = []
    for cls_idx, _ in sorted_by_acc_desc:
        mask = all_true == cls_idx
        wrong = (mask & (all_pred != cls_idx)).sum()

        if wrong >= 5:
            best_3.append(cls_idx)

        if len(best_3) == 3:
            break

    chosen_classes = best_3 + worst_3

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("\nChosen classes:")
    for cls_idx in chosen_classes:
        tag = "BEST" if cls_idx in best_3 else "WORST"
        print(f"  [{tag}] {idx_to_class[cls_idx]} — acc={per_class_acc[cls_idx]:.3f}")

    return chosen_classes, best_3, all_true, all_pred, all_conf, all_paths


def select_images(cls_idx, all_true, all_pred, all_conf):
    mask = all_true == cls_idx

    correct_mask = mask & (all_pred == all_true)
    correct_confs = all_conf.copy()
    correct_confs[~correct_mask] = -1
    correct_top5_idx = np.argsort(correct_confs)[::-1][:5]

    wrong_mask = mask & (all_pred != all_true)
    wrong_confs = all_conf.copy()
    wrong_confs[~wrong_mask] = -1
    wrong_top5_idx = np.argsort(wrong_confs)[::-1][:5]

    return correct_top5_idx, wrong_top5_idx


def gradcam_vis(
    model,
    chosen_classes,
    best_3,
    all_true,
    all_pred,
    all_conf,
    all_paths,
    idx_to_class,
    val_transform,
):

    target_layer = [model.layer4.block2]

    def load_rgb(img_path):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0

    def path_to_tensor(img_path):
        img = Image.open(img_path).convert("RGB")
        return val_transform(img).unsqueeze(0).to(device)

    SAVE_PATH = "Grad_Cam_Vis"
    os.makedirs(SAVE_PATH, exist_ok=True)

    with GradCAM(model=model, target_layers=target_layer) as cam:
        for cls_idx in chosen_classes:
            cls_name = idx_to_class[cls_idx]
            tag = "BEST" if cls_idx in best_3 else "WORST"
            cls_dir = os.path.join(SAVE_PATH, f"{tag}_{cls_name}.png")

            correct_idx, wrong_idx = select_images(
                cls_idx, all_true, all_pred, all_conf
            )

            print("\n" + "=" * 60)
            print(f"Class: {cls_name}  |  Category: {tag.upper()}")
            print("=" * 60)

            correct_images = []
            wrong_images = []
            wrong_pred_names = []

            for rank, sample_idx in enumerate(correct_idx, start=1):
                img_path = all_paths[sample_idx]
                rgb_img = load_rgb(img_path)
                tensor = path_to_tensor(img_path)

                targets = [ClassifierOutputTarget(all_true[sample_idx])]
                grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
                vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                correct_images.append(vis)

            for rank, sample_idx in enumerate(wrong_idx, start=1):
                img_path = all_paths[sample_idx]
                rgb_img = load_rgb(img_path)
                tensor = path_to_tensor(img_path)

                pred_class = all_pred[sample_idx]
                targets = [ClassifierOutputTarget(all_true[sample_idx])]
                grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
                vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                wrong_images.append(vis)
                wrong_pred_names.append(idx_to_class[pred_class])

            n_cols = max(len(correct_images), len(wrong_images))
            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

            if n_cols == 1:
                axes = axes.reshape(2, 1)

            for i in range(n_cols):
                ax = axes[0, i]
                if i < len(correct_images):
                    ax.imshow(correct_images[i])
                ax.set_title(f"Correct {i+1}")
                ax.axis("off")

            for i in range(n_cols):
                ax = axes[1, i]
                if i < len(wrong_images):
                    ax.imshow(wrong_images[i])
                ax.set_title(f"Incorrect {i+1}\nPred: {wrong_pred_names[i]}")
                ax.axis("off")

            fig.suptitle(
                f"GradCAM Visualization\nClass: {cls_name} ({tag.upper()})",
                fontsize=20,
                fontweight="bold",
            )

            plt.tight_layout()
            plt.savefig(cls_dir)
            plt.close()

            print(f"Saved to {cls_dir}")

    print("\nDone. Results saved to:", SAVE_PATH)


if __name__ == "__main__":

    val_path = "val"
    classes = sorted(os.listdir(val_path))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    val_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = in100_dataset(val_path, classes, class_to_idx, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    exp = "BIN"
    ckpt = torch.load(f"saved_models/{exp}.pth", map_location=device)

    model = resnet18(num_classes=num_classes, norm_layer=batch_instance_norm).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    chosen_classes, best_3, all_true, all_pred, all_conf, all_paths = (
        select_best_worst_classes(model, val_loader, class_to_idx)
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    gradcam_vis(
        model,
        chosen_classes,
        best_3,
        all_true,
        all_pred,
        all_conf,
        all_paths,
        idx_to_class,
        val_transform,
    )
