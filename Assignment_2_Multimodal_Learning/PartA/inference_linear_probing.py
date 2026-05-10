"""
inference_linear_probing.py — Auto-grader Part A linear-probe inference.

Auto-grader CLI (must be obeyed exactly):

  python inference_linear_probing.py \
      --model_type   <clip|dino_student|dino_teacher> \
      --model_dir    <path_to_partA_models_directory> \
      --pooling_type <cls|gap> \
      --probe_task   <count|color> \
      --data_path    <path_to_probe_json> \
      --output_file  <path_to_output_json>

Files expected inside --model_dir:
  clip.pt          CLIP checkpoint (full model state-dict, our format)
  dino.pt          DINO checkpoint  (only needed for dino_student / dino_teacher)
  tokenizer.json   CLEVRTokenizer JSON dump (only needed for clip)
  probes.pt        Pre-trained linear-probe weights produced by linear_probe.py

Output JSON (per the submission guidelines):
  count : {image_filename: <int predicted count>}
  color : {image_filename: [<list of predicted color strings>]}
"""

import os
import json
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from clip_model import CLIP
from dino_model import DINO
from dataset    import get_clip_probe_transform
from tokenizer  import CLEVRTokenizer


# ──────────────────────────── Dataset ───────────────────────────────────── #

class ProbeImageDataset(Dataset):
    """
    Loads images using the absolute `image_path` field provided by the
    auto-grader. Falls back to `image_filename` only if `image_path` is missing.
    """
    def __init__(self, examples: List[dict], transform):
        self.examples  = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex   = self.examples[idx]
        path = ex.get('image_path') or ex['image_filename']
        img  = Image.open(path).convert('RGB')
        return self.transform(img), ex['image_filename']


# ──────────────────────────── Model loading ─────────────────────────────── #

def _strip_compile_prefix(sd: dict) -> dict:
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def _patch_legacy_clip(sd: dict) -> dict:
    """Older CLIP ckpts have no `gap_proj.*`; mirror `image_proj` so the strict
    state-dict load succeeds. The probe only uses `image_encoder`, so this is
    semantically a no-op."""
    if 'gap_proj.weight' not in sd and 'image_proj.weight' in sd:
        sd['gap_proj.weight'] = sd['image_proj.weight'].clone()
        sd['gap_proj.bias']   = sd['image_proj.bias'].clone()
    return sd


def _infer_dino_out_dim(sd: dict) -> int:
    c = sd.get('center')
    return int(c.shape[1]) if (c is not None and c.ndim == 2) else 4096


def _load_clip(model_dir: str, device) -> CLIP:
    tok = CLEVRTokenizer.load(os.path.join(model_dir, 'tokenizer.json'))
    model = CLIP(vocab_size=tok.vocab_size).to(device)
    ckpt = torch.load(os.path.join(model_dir, 'clip.pt'),
                      map_location=device, weights_only=False)
    sd   = _patch_legacy_clip(_strip_compile_prefix(ckpt.get('model', ckpt)))
    model.load_state_dict(sd)
    model.eval()
    return model


def _load_dino(model_dir: str, device) -> DINO:
    ckpt = torch.load(os.path.join(model_dir, 'dino.pt'),
                      map_location=device, weights_only=False)
    sd   = _strip_compile_prefix(ckpt.get('model', ckpt))
    model = DINO(out_dim=_infer_dino_out_dim(sd)).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def get_backbone(model_type: str, model_dir: str, device):
    if model_type == 'clip':
        return _load_clip(model_dir, device).image_encoder
    dino = _load_dino(model_dir, device)
    if model_type == 'dino_student':
        return dino.student.backbone
    if model_type == 'dino_teacher':
        return dino.teacher.backbone
    raise ValueError(f'Unknown --model_type: {model_type}')


# ──────────────────────────── Feature extraction ────────────────────────── #

@torch.no_grad()
def extract_features(backbone, loader, device, use_cls: bool, tta: bool):
    backbone.eval()
    feats, fnames = [], []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        out  = backbone(imgs)
        f    = out[:, 0] if use_cls else out[:, 1:].mean(1)

        if tta:
            out2 = backbone(torch.flip(imgs, dims=[-1]))
            f2   = out2[:, 0] if use_cls else out2[:, 1:].mean(1)
            f    = F.normalize(f, dim=-1) + F.normalize(f2, dim=-1)

        feats.append(F.normalize(f.float(), dim=-1).cpu())
        fnames.extend(names)
    return torch.cat(feats, dim=0), fnames


# ──────────────────────────── Main ──────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type',   required=True,
                   choices=['clip', 'dino_student', 'dino_teacher'])
    p.add_argument('--model_dir',    required=True)
    p.add_argument('--pooling_type', required=True, choices=['cls', 'gap'])
    p.add_argument('--probe_task',   required=True, choices=['count', 'color'])
    p.add_argument('--data_path',    required=True)
    p.add_argument('--output_file',  required=True)
    p.add_argument('--batch_size',   type=int, default=256)
    p.add_argument('--num_workers',  type=int, default=8)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Read input JSON ─────────────────────────────────────────────────── #
    with open(args.data_path) as f:
        raw = json.load(f)
    examples = raw['examples'] if (isinstance(raw, dict) and 'examples' in raw) else raw

    # ── Load probe weights and metadata for this (model, pool, task) ─────── #
    probes = torch.load(os.path.join(args.model_dir, 'probes.pt'),
                        map_location='cpu', weights_only=False)
    info = probes[args.model_type][args.pooling_type][args.probe_task]
    resolution = int(info.get('resolution', 288))
    use_tta    = bool(info.get('tta', True))

    # ── Load encoder ───────────────────────────────────────────────────── #
    backbone = get_backbone(args.model_type, args.model_dir, device)

    # ── Image pipeline ─────────────────────────────────────────────────── #
    transform = get_clip_probe_transform(size=resolution)
    ds = ProbeImageDataset(examples, transform)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Extract features ──────────────────────────────────────────────── #
    feats, filenames = extract_features(
        backbone, loader, device,
        use_cls=(args.pooling_type == 'cls'), tta=use_tta,
    )

    # ── Build linear probe and apply ──────────────────────────────────── #
    probe = nn.Linear(int(info['feat_dim']),
                      info['weight'].shape[0]).to(device)
    probe.weight.data.copy_(info['weight'].to(device))
    probe.bias.data.copy_(info['bias'].to(device))
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device)).cpu()

    # ── Assemble output JSON in the schema mandated by the guidelines ──── #
    output = {}
    if args.probe_task == 'count':
        min_count = int(info['min_count'])
        preds = logits.argmax(dim=1) + min_count
        for fn, c in zip(filenames, preds.tolist()):
            output[fn] = int(c)
    else:
        colors = info['colors']
        mask = (torch.sigmoid(logits) >= 0.5)
        for fn, m in zip(filenames, mask.tolist()):
            output[fn] = [c for c, b in zip(colors, m) if b]

    # ── Write ──────────────────────────────────────────────────────────── #
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Wrote {len(output)} predictions to {args.output_file}')


if __name__ == '__main__':
    main()
