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
from transformers import CLIPTextModel, CLIPTokenizer

from model import vae, conditional_unet, diffusion_schedule

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

SAVE_DIR = "LDM_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = "A2_dataset/Part_A"
VAE_CKPT = "VAE_checkpoints/vae_best.pth"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_SAVE_DIR = os.path.join(SAVE_DIR, "clip_encoder")

T_STEPS = 500
CFG_DROPOUT = 0.1
GUIDANCE_SCALE = 4.0
MODEL_CHANNELS = 256
NUM_HEADS = 8
CONTEXT_DIM = 512


# CLEVR dataset with captions for LDM training
class clevr_ldm(Dataset):
    def __init__(self, split_dir, transform, split):
        print(split_dir)
        self.img_dir = os.path.join(split_dir, "images")
        self.transform = transform

        captions_file = os.path.join(split_dir, f"clevr_{split}_captions.json")

        with open(captions_file, "r") as f:
            captions_data = json.load(f)

        self.image_names = sorted(os.listdir(self.img_dir))
        self.all_captions = self.parse(captions_data)

    def parse(self, data):
        return {item["image_filename"]: item["caption"] for item in data}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        file = self.image_names[idx]
        image = Image.open(os.path.join(self.img_dir, file)).convert("RGB")

        caption = self.all_captions[file]

        return self.transform(image), caption


def compute_latent_stats(vae_model, dataloader, device):
    vae_model.eval()
    all_latents = []
    total = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing latent stats"):
            mu, _ = vae_model.encode(images.to(device))
            all_latents.append(mu.cpu())
            total += images.shape[0]

    all_latents = torch.cat(all_latents, dim=0)
    mean = all_latents.mean()
    std = all_latents.std()
    print(f"Latent stats - mean: {mean:.4f}, std: {std:.4f}")
    return mean, std


def plot_metrics(train_history, val_history):
    plt.figure(figsize=(7, 5))
    plt.plot([m["loss"] for m in train_history], label="Train Loss")
    plt.plot([m["loss"] for m in val_history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LDM Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss.png"))
    plt.close()


def sample_images(
    model,
    schedule,
    vae_model,
    clip_text_model,
    tokenizer,
    prompts,
    latent_mean,
    latent_std,
    device,
    guidance_scale=GUIDANCE_SCALE,
):
    model.eval()
    B = len(prompts)

    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    context = clip_text_model(**tokens).last_hidden_state
    null_ctx = model.null_embedding.expand(B, -1, -1)

    zt = torch.randn(B, 4, 16, 16, device=device)

    with torch.no_grad():
        for t_idx in tqdm(
            reversed(range(schedule.T)), desc="Sampling", total=schedule.T, leave=False
        ):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            eps_cond = model(zt, t_batch, context)
            eps_uncond = model(zt, t_batch, null_ctx)
            eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

            beta_t = schedule.betas[t_idx]
            alpha_t = schedule.alphas[t_idx]
            sqrt_1m_acp = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            mean = (1.0 / alpha_t.sqrt()) * (zt - (beta_t / sqrt_1m_acp) * eps)

            if t_idx > 0:
                zt = mean + beta_t.sqrt() * torch.randn_like(zt)
            else:
                zt = mean

    z0 = zt * latent_std.to(device) + latent_mean.to(device)
    images = vae_model.decode(z0)
    return (images.clamp(-1, 1) + 1) / 2


def compute_lpips_metric(img1, img2):
    diff = (img1 - img2).pow(2).mean(dim=0)
    return diff.mean().item()


def save_random_reconstructions(
    model,
    schedule,
    vae_model,
    clip_text_model,
    tokenizer,
    val_loader,
    latent_mean,
    latent_std,
    device,
    save_dir,
):
    recon_dir = os.path.join(save_dir, "random_reconstructions")
    os.makedirs(recon_dir, exist_ok=True)
    print("Generating 10 random validation samples")

    model.eval()

    images, captions = next(iter(val_loader))
    images = images.to(device)

    indices = np.random.choice(len(images), size=min(10, len(images)), replace=False)

    selected_images = images[indices]
    selected_captions = [captions[i] for i in indices]

    with torch.no_grad():
        gen_images = sample_images(
            model,
            schedule,
            vae_model,
            clip_text_model,
            tokenizer,
            selected_captions,
            latent_mean,
            latent_std,
            device,
        )

        orig = (selected_images.clamp(-1, 1) + 1) / 2

        for i in range(len(selected_images)):
            o = orig[i].cpu().permute(1, 2, 0).numpy()
            g = gen_images[i].cpu().permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(o)
            axes[0].axis("off")
            axes[0].set_title("Original")
            axes[1].imshow(g)
            axes[1].axis("off")
            axes[1].set_title("Generated")
            fig.suptitle(selected_captions[i], fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(recon_dir, f"sample_{i+1}.png"))
            plt.close()


def train_ldm(epochs):

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("Loading CLIP text encoder")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    clip_text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_text_model.eval()

    os.makedirs(CLIP_SAVE_DIR, exist_ok=True)

    clip_text_model.save_pretrained(CLIP_SAVE_DIR)
    tokenizer.save_pretrained(CLIP_SAVE_DIR)

    print(f"CLIP encoder saved to {CLIP_SAVE_DIR}")

    for p in clip_text_model.parameters():
        p.requires_grad = False

    print("Preparing datasets")
    train_dataset = clevr_ldm(os.path.join(DATA_DIR, "train"), transform, "train")
    val_dataset = clevr_ldm(os.path.join(DATA_DIR, "val"), transform, "val")

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

    print("Loading pretrained VAE")
    vae_model = vae().to(device)
    vae_model.load_state_dict(
        torch.load(VAE_CKPT, map_location=device, weights_only=True)
    )
    vae_model.eval()

    for p in vae_model.parameters():
        p.requires_grad = False

    stats_path = os.path.join(SAVE_DIR, "latent_stats.pt")
    if os.path.exists(stats_path):
        print("Loading cached latent statistics")
        stats = torch.load(stats_path)
        latent_mean, latent_std = stats["mean"].to(device), stats["std"].to(device)
    else:
        print("Computing latent statistics from training set")
        latent_mean, latent_std = compute_latent_stats(vae_model, train_loader, device)
        torch.save({"mean": latent_mean.cpu(), "std": latent_std.cpu()}, stats_path)
        latent_mean, latent_std = latent_mean.to(device), latent_std.to(device)

    schedule = diffusion_schedule(T_STEPS, device)

    print("Initializing U-Net LDM")
    model = conditional_unet().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_model_path = os.path.join(SAVE_DIR, "ldm_best.pth")

    print("Starting training")
    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs} - Training")
        model.train()
        total_loss = 0.0

        for images, captions in tqdm(
            train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"
        ):
            images = images.to(device)
            B = images.shape[0]

            with torch.no_grad():
                tokens = tokenizer(
                    list(captions),
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                context = clip_text_model(**tokens).last_hidden_state
                mu, logvar = vae_model.encode(images)

                z0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                z0_norm = (z0 - latent_mean) / latent_std

                t = torch.randint(0, T_STEPS, (B,), device=device, dtype=torch.long)
                zt, noise = schedule.q_sample(z0_norm, t)

            null_ctx = model.null_embedding.expand(B, -1, -1)
            mask = (torch.rand(B, device=device) < CFG_DROPOUT)[:, None, None]
            context = torch.where(mask, null_ctx, context)

            eps_pred = model(zt, t, context)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_metrics = {"loss": total_loss / len(train_loader)}
        train_history.append(train_metrics)

        print(f"Epoch {epoch+1}/{epochs} - Validation")
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, captions in tqdm(
                val_loader, desc=f"Val   Epoch {epoch+1}/{epochs}"
            ):
                images = images.to(device)
                B = images.shape[0]

                tokens = tokenizer(
                    list(captions),
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                context = clip_text_model(**tokens).last_hidden_state
                mu, logvar = vae_model.encode(images)
                z0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                z0_norm = (z0 - latent_mean) / latent_std

                t = torch.randint(0, T_STEPS, (B,), device=device, dtype=torch.long)
                zt, noise = schedule.q_sample(z0_norm, t)

                eps_pred = model(zt, t, context)
                total_val_loss += F.mse_loss(eps_pred, noise).item()

        val_metrics = {"loss": total_val_loss / len(val_loader)}
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_metrics['loss']:.5f} | "
            f"Val Loss={val_metrics['loss']:.5f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.5f}"
            )

    print("Training complete")

    with open(os.path.join(SAVE_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_history, f, indent=4)
    with open(os.path.join(SAVE_DIR, "val_metrics.json"), "w") as f:
        json.dump(val_history, f, indent=4)

    plot_metrics(train_history, val_history)

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    save_random_reconstructions(
        model,
        schedule,
        vae_model,
        clip_text_model,
        tokenizer,
        val_loader,
        latent_mean,
        latent_std,
        device,
        SAVE_DIR,
    )
    print("Random Reconstructions saved")


if __name__ == "__main__":
    print("Training LDM")
    train_ldm(epochs=100)
