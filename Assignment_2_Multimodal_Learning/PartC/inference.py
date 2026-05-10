import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from model import vae, conditional_unet, diffusion_schedule, T_STEPS, GUIDANCE_SCALE

RECON_BATCH_SIZE = 512
GEN_BATCH_SIZE = 256
NUM_WORKERS = 8


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_to_pil(t):
    arr = (t.permute(1, 2, 0).cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


img_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def resolve_path(model_dir, filename):
    direct = os.path.join(model_dir, filename)
    if os.path.exists(direct):
        return direct
    nested = os.path.join(model_dir, "checkpoints", filename)
    if os.path.exists(nested):
        return nested
    raise FileNotFoundError(
        f"{filename} not found in {model_dir} or {model_dir}/checkpoints"
    )


def load_vae(model_dir, device):
    model = vae().to(device)
    model.load_state_dict(
        torch.load(
            resolve_path(model_dir, "vae_best.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model


def load_ldm(model_dir, device):
    model = conditional_unet().to(device)
    model.load_state_dict(
        torch.load(
            resolve_path(model_dir, "ldm_best.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    return model


def load_latent_stats(model_dir, device):
    stats = torch.load(
        resolve_path(model_dir, "latent_stats.pt"),
        map_location="cpu",
        weights_only=True,
    )
    return stats["mean"].to(device), stats["std"].to(device)


def load_clip(model_dir, device):
    clip_dir = os.path.join(model_dir, "clip_encoder")
    if not os.path.isdir(clip_dir):
        clip_dir = os.path.join(model_dir, "checkpoints", "clip_encoder")

    tokenizer = CLIPTokenizer.from_pretrained(clip_dir, local_files_only=True)
    clip_text_model = CLIPTextModel.from_pretrained(clip_dir, local_files_only=True).to(
        device
    )

    clip_text_model.eval()
    for p in clip_text_model.parameters():
        p.requires_grad = False
    return tokenizer, clip_text_model


class recon_dataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        return load_image(str(p), img_transform), p.stem


def run_reconstruct(model_dir, data_path, output_dir, device):

    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(
        [p for p in Path(data_path).iterdir() if p.suffix.lower() in valid_exts]
    )
    if not img_paths:
        raise ValueError(f"No images found in {data_path}")

    vae_model = load_vae(model_dir, device)
    loader = DataLoader(
        recon_dataset(img_paths),
        batch_size=RECON_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for imgs, stems in tqdm(loader, desc="Reconstructing"):
        imgs = imgs.to(device)

        with torch.no_grad():
            mu, _ = vae_model.encode(imgs)
            recons = vae_model.decode(mu)

        for recon_t, stem in zip(recons, stems):
            tensor_to_pil(recon_t).save(os.path.join(output_dir, stem + ".png"))


def ddpm_sample(model, schedule, context, latent_mean, latent_std, device):
    B = context.shape[0]
    null_ctx = model.null_embedding.expand(B, -1, -1)
    zt = torch.randn(B, 4, 16, 16, device=device)

    with torch.no_grad():
        for t_idx in tqdm(
            reversed(range(schedule.T)), desc="Sampling", total=schedule.T, leave=False
        ):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            eps_cond = model(zt, t_batch, context)
            eps_uncond = model(zt, t_batch, null_ctx)
            eps = (1 + GUIDANCE_SCALE) * eps_cond - GUIDANCE_SCALE * eps_uncond

            beta_t = schedule.betas[t_idx]
            alpha_t = schedule.alphas[t_idx]
            sqrt_1m_acp = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            mean = (1.0 / alpha_t.sqrt()) * (zt - (beta_t / sqrt_1m_acp) * eps)

            if t_idx > 0:
                zt = mean + beta_t.sqrt() * torch.randn_like(zt)
            else:
                zt = mean

    return zt * latent_std + latent_mean


def run_generate(model_dir, data_path, output_dir, device):

    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r") as f:
        entries = json.load(f)

    vae_model = load_vae(model_dir, device)
    ldm_model = load_ldm(model_dir, device)
    latent_mean, latent_std = load_latent_stats(model_dir, device)
    tokenizer, clip_text_model = load_clip(model_dir, device)
    schedule = diffusion_schedule(T_STEPS, device)

    for batch_start in tqdm(range(0, len(entries), GEN_BATCH_SIZE), desc="Generating"):
        batch = entries[batch_start : batch_start + GEN_BATCH_SIZE]
        captions = [e["caption"] for e in batch]
        filenames = [e["image_filename"] for e in batch]

        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            context = clip_text_model(**tokens).last_hidden_state

        z0 = ddpm_sample(ldm_model, schedule, context, latent_mean, latent_std, device)

        with torch.no_grad():
            imgs = vae_model.decode(z0).clamp(-1, 1)

        for img_t, fname in zip(imgs, filenames):
            tensor_to_pil(img_t).save(os.path.join(output_dir, fname))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--task", required=True, choices=["reconstruct", "generate"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"Using device: {device}")

    if args.task == "reconstruct":
        run_reconstruct(args.model_dir, args.data_path, args.output_dir, device)
    elif args.task == "generate":
        run_generate(args.model_dir, args.data_path, args.output_dir, device)


if __name__ == "__main__":
    main()
