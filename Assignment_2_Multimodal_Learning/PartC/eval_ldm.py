import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
import shutil
import tempfile
from cleanfid import fid

from model import vae, conditional_unet, diffusion_schedule
from train_ldm import clevr_ldm

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
LDM_CKPT = "LDM_checkpoints/ldm_best.pth"
LATENT_STATS = "LDM_checkpoints/latent_stats.pt"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

T_STEPS = 500
GUIDANCE_SCALE = 4.0


# Convert tensor to PIL Image
def tensor_to_pil(t):
    arr = t.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# Save real images from dataloader and return captions
def save_real_images(loader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    captions_list = []

    for batch_images, batch_captions in tqdm(loader, desc="Saving real images"):
        for img, caption in zip(batch_images, batch_captions):
            img = (img + 1) / 2
            tensor_to_pil(img).save(os.path.join(out_dir, f"{idx:06d}.png"))
            captions_list.append(caption)
            idx += 1

    return captions_list


# Generate images from text prompts using diffusion
def sample_images_batch(
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
        for t_idx in reversed(range(schedule.T)):
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


# Generate all images for given captions and save to directory
def generate_all_images(
    model,
    schedule,
    vae_model,
    clip_text_model,
    tokenizer,
    captions,
    latent_mean,
    latent_std,
    device,
    out_dir,
    batch_size=16,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idx = 0
    num_batches = (len(captions) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating images"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(captions))
        batch_captions = captions[start_idx:end_idx]

        gen_images = sample_images_batch(
            model,
            schedule,
            vae_model,
            clip_text_model,
            tokenizer,
            batch_captions,
            latent_mean,
            latent_std,
            device,
        )

        for img in gen_images:
            tensor_to_pil(img).save(os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1


# Compute CLIP similarity between images and their captions
def compute_clip_similarity(images, captions, clip_model, clip_processor, device):
    all_similarities = []

    for img_tensor, caption in tqdm(
        zip(images, captions), total=len(images), desc="Computing CLIP similarity"
    ):
        img_pil = tensor_to_pil(img_tensor)

        inputs = clip_processor(
            text=[caption], images=img_pil, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image.item()

        all_similarities.append(similarity)

    return torch.tensor(all_similarities)


# Compute FID score between real and generated images
def compute_fid_score(real_dir, gen_dir):
    print(f"\nComputing FID score...")
    score = fid.compute_fid(
        fdir1=real_dir,
        fdir2=gen_dir,
        mode="clean",
        num_workers=8,
        batch_size=256,
        device=device,
    )
    print(f"Validation FID: {score:.4f}")
    return score


# Save comparison pairs of real and generated images
def save_comparison_pairs(
    real_images, gen_images, captions, similarities, save_dir, prefix, indices
):
    for i, idx in enumerate(indices):
        real_img = tensor_to_pil(real_images[idx])
        gen_img = tensor_to_pil(gen_images[idx])

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(real_img)
        axes[0].axis("off")
        axes[0].set_title("Original")
        axes[1].imshow(gen_img)
        axes[1].axis("off")
        axes[1].set_title(f"Generated (CLIP: {similarities[idx]:.2f})")
        fig.suptitle(captions[idx], fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{i+1}.png"))
        plt.close()


# Main evaluation function
def evaluate_ldm():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("Loading CLIP models")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    clip_text_model = CLIPTextModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_text_model.eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model.eval()

    for p in clip_text_model.parameters():
        p.requires_grad = False
    for p in clip_model.parameters():
        p.requires_grad = False

    print("Loading validation dataset")
    val_dataset = clevr_ldm(os.path.join(DATA_DIR, "val"), transform, "val")
    print(f"Val samples: {len(val_dataset)}")

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

    print("Loading latent statistics")
    stats = torch.load(LATENT_STATS, map_location=device, weights_only=True)
    latent_mean = stats["mean"].to(device)
    latent_std = stats["std"].to(device)

    print("Loading pretrained LDM")
    schedule = diffusion_schedule(T_STEPS, device)
    ldm_model = conditional_unet().to(device)
    ldm_model.load_state_dict(
        torch.load(LDM_CKPT, map_location=device, weights_only=True)
    )
    ldm_model.eval()

    print("\nStarting evaluation...")

    tmp_root = tempfile.mkdtemp(prefix="ldm_eval_")

    real_dir = os.path.join(tmp_root, "real_val")
    gen_dir = os.path.join(tmp_root, "gen_val")

    try:
        print("\nSaving real validation images")
        captions = save_real_images(val_loader, real_dir)

        print("\nGenerating images from captions")
        generate_all_images(
            ldm_model,
            schedule,
            vae_model,
            clip_text_model,
            tokenizer,
            captions,
            latent_mean,
            latent_std,
            device,
            gen_dir,
            batch_size=16,
        )

        val_fid = compute_fid_score(real_dir, gen_dir)

        print(f"\n===== FINAL FID =====")
        print(f"Validation FID: {val_fid:.4f}")

        fid_results = {"val_fid": val_fid}

        with open(os.path.join(SAVE_DIR, "fid_results.json"), "w") as f:
            json.dump(fid_results, f, indent=4)

        print("\nLoading images for CLIP similarity comparison")
        all_real_images = []
        all_gen_images = []

        for i in tqdm(range(len(captions)), desc="Loading images"):
            real_img = Image.open(os.path.join(real_dir, f"{i:06d}.png"))
            gen_img = Image.open(os.path.join(gen_dir, f"{i:06d}.png"))

            real_tensor = transform(real_img)
            gen_tensor = transforms.ToTensor()(gen_img)

            all_real_images.append((real_tensor + 1) / 2)
            all_gen_images.append(gen_tensor)

        print("\nComputing CLIP similarities")
        similarities = compute_clip_similarity(
            all_gen_images, captions, clip_model, clip_processor, device
        )

        sorted_idx = torch.argsort(similarities, descending=True)
        best_idx = sorted_idx[:10]
        worst_idx = sorted_idx[-10:]

        recon_dir = os.path.join(SAVE_DIR, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)

        print("\nSaving best reconstructions")
        save_comparison_pairs(
            all_real_images,
            all_gen_images,
            captions,
            similarities,
            recon_dir,
            "best",
            best_idx,
        )

        print("Saving worst reconstructions")
        save_comparison_pairs(
            all_real_images,
            all_gen_images,
            captions,
            similarities,
            recon_dir,
            "worst",
            worst_idx,
        )

        print(f"\nBest CLIP similarity: {similarities[best_idx[0]]:.4f}")
        print(f"Worst CLIP similarity: {similarities[worst_idx[-1]]:.4f}")
        print(f"Mean CLIP similarity: {similarities.mean():.4f}")

        clip_stats = {
            "best_similarity": similarities[best_idx[0]].item(),
            "worst_similarity": similarities[worst_idx[-1]].item(),
            "mean_similarity": similarities.mean().item(),
            "std_similarity": similarities.std().item(),
        }

        with open(os.path.join(SAVE_DIR, "clip_stats.json"), "w") as f:
            json.dump(clip_stats, f, indent=4)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nEvaluation complete!")
    print(f"Results saved to {SAVE_DIR}")


if __name__ == "__main__":
    print("Evaluating LDM")
    evaluate_ldm()
