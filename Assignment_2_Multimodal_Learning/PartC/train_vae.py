import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import shutil
import tempfile
from cleanfid import fid

from model import vae

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


SAVE_DIR = "VAE_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# CLEVR dataset class for VAE training
class clevr_vae(Dataset):
    def __init__(self, split_dir, transform):
        self.img_dir = os.path.join(split_dir, "images")
        self.transform = transform

        self.image_filenames = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        return self.transform(image)


# VAE loss function combining reconstruction loss and KL divergence
def vae_loss(recon, x, mu, logvar, kl_weight=1e-6):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss

    return total, recon_loss.item(), kl_loss.item()


def plot_metrics(train_history, val_history):
    for metric in ["loss", "recon_loss", "kl_loss"]:
        train_vals = [m[metric] for m in train_history]
        val_vals = [m[metric] for m in val_history]

        plt.figure(figsize=(7, 5))
        plt.plot(train_vals, label=f"Train {metric}")
        plt.plot(val_vals, label=f"Val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} over epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{metric}.png"))
        plt.close()


def save_reconstructions(model, val_loader, device, save_dir):
    recon_dir = os.path.join(save_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    model.eval()
    all_losses = []
    all_images = []
    all_recons = []

    with torch.no_grad():
        for images in tqdm(val_loader, desc="Saving reconstructions"):
            images = images.to(device)
            recon, mu, logvar = model(images)
            losses = F.mse_loss(recon, images, reduction="none").mean(dim=[1, 2, 3])

            all_losses.append(losses.cpu())
            all_images.append(images.cpu())
            all_recons.append(recon.cpu())

    all_losses = torch.cat(all_losses)
    all_images = torch.cat(all_images)
    all_recons = torch.cat(all_recons)

    sorted_idx = torch.argsort(all_losses)
    best_idx = sorted_idx[:10]
    worst_idx = sorted_idx[-10:]

    def save_pair(idx, filename):
        orig = (all_images[idx].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        recon = (all_recons[idx].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(orig)
        axes[0].axis("off")
        axes[0].set_title("Original")
        axes[1].imshow(recon)
        axes[1].axis("off")
        axes[1].set_title(f"Recon ({all_losses[idx]:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, filename))
        plt.close()

    for i, idx in enumerate(best_idx):
        save_pair(idx, f"best_{i+1}.png")
    for i, idx in enumerate(worst_idx):
        save_pair(idx, f"worst_{i+1}.png")


def tensor_to_pil(t: torch.Tensor):
    arr = (t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def save_real_images(loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for batch in tqdm(loader, desc=f"Saving real {os.path.basename(out_dir)}"):
        for img in batch:
            tensor_to_pil(img).save(os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1


def reconstruct_images(model, loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Reconstructing images"):
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            recons = model.decode(z)

            for img in recons:
                tensor_to_pil(img).save(os.path.join(out_dir, f"{saved:06d}.png"))
                saved += 1


def compute_fid_score(real_dir, gen_dir, split_name):
    print(f"\n[FID] Computing {split_name} FID...")
    score = fid.compute_fid(
        fdir1=real_dir,
        fdir2=gen_dir,
        mode="clean",
        num_workers=8,
        batch_size=256,
        device=device,
    )
    print(f"{split_name} FID: {score:.4f}")
    return score


def train_vae(epochs):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    data_dir = "A2_dataset/Part_A"

    print("Preparing datasets")
    train_dataset = clevr_vae(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = clevr_vae(os.path.join(data_dir, "val"), transform=transform)

    print(f"Train samples: {len(train_dataset)}" f" | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print("Initializing model and optimizer")
    model = vae().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_model_path = os.path.join(SAVE_DIR, "vae_best.pth")

    print("Starting training")
    for epoch in range(epochs):

        model.train()
        total_loss = total_recon = total_kl = 0.0

        for images in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_l, kl_l = vae_loss(recon, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l
            total_kl += kl_l

        scheduler.step()
        n = len(train_loader)
        train_metrics = {
            "loss": total_loss / n,
            "recon_loss": total_recon / n,
            "kl_loss": total_kl / n,
        }
        train_history.append(train_metrics)

        model.eval()
        total_loss = total_recon = total_kl = 0.0

        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Val   Epoch {epoch+1}/{epochs}"):
                images = images.to(device)
                recon, mu, logvar = model(images)
                loss, recon_l, kl_l = vae_loss(recon, images, mu, logvar)

                total_loss += loss.item()
                total_recon += recon_l
                total_kl += kl_l

        n = len(val_loader)
        val_metrics = {
            "loss": total_loss / n,
            "recon_loss": total_recon / n,
            "kl_loss": total_kl / n,
        }
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_metrics['loss']:.5f}  "
            f"Recon={train_metrics['recon_loss']:.5f}  "
            f"KL={train_metrics['kl_loss']:.2f} | "
            f"Val Loss={val_metrics['loss']:.5f}  "
            f"Recon={val_metrics['recon_loss']:.5f}  "
            f"KL={val_metrics['kl_loss']:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.5f}"
            )

    print("\nTraining complete")

    with open(os.path.join(SAVE_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_history, f, indent=4)
    with open(os.path.join(SAVE_DIR, "val_metrics.json"), "w") as f:
        json.dump(val_history, f, indent=4)

    plot_metrics(train_history, val_history)
    print(f"\nBest Val Loss: {best_val_loss:.5f}")

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    print("\nComputing FID scores (Val first, then Train)...")

    tmp_root = tempfile.mkdtemp(prefix="vae_fid_")

    dirs = {
        "real_train": os.path.join(tmp_root, "real_train"),
        "gen_train": os.path.join(tmp_root, "gen_train"),
        "real_val": os.path.join(tmp_root, "real_val"),
        "gen_val": os.path.join(tmp_root, "gen_val"),
    }

    try:
        print("\n[Val] Saving real images")
        save_real_images(val_loader, dirs["real_val"])

        print("[Val] Generating reconstructions")
        reconstruct_images(model, val_loader, dirs["gen_val"])

        val_fid = compute_fid_score(dirs["real_val"], dirs["gen_val"], "Val")
        print(f"Val   FID: {val_fid:.4f}")

        print("\n[Train] Saving real images")
        save_real_images(train_loader, dirs["real_train"])

        print("[Train] Generating reconstructions")
        reconstruct_images(model, train_loader, dirs["gen_train"])

        train_fid = compute_fid_score(dirs["real_train"], dirs["gen_train"], "Train")

        print("\n===== FINAL FID =====")
        print(f"Val   FID: {val_fid:.4f}")
        print(f"Train FID: {train_fid:.4f}")

        fid_results = {
            "train_fid": train_fid,
            "val_fid": val_fid,
        }

        with open(os.path.join(SAVE_DIR, "fid_results.json"), "w") as f:
            json.dump(fid_results, f, indent=4)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    save_reconstructions(model, val_loader, device, SAVE_DIR)
    print("Reconstructions saved")


if __name__ == "__main__":
    print("Training VAE")
    train_vae(epochs=100)
