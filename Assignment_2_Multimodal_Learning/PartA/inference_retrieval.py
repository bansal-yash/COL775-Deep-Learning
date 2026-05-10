"""
inference_retrieval.py — Auto-grader Part A CLIP retrieval inference.

Auto-grader CLI (must be obeyed exactly):

  python inference_retrieval.py \
      --model_type    clip \
      --model_dir     <path_to_partA_models_directory> \
      --retrieval_task <i2t|t2i> \
      --data_path     <path_to_caption_json> \
      --output_file   <path_to_output_json>

Files expected inside --model_dir:
  clip.pt          CLIP checkpoint (full model state-dict, our format)
  tokenizer.json   CLEVRTokenizer JSON dump

Output JSON (per the submission guidelines):
  i2t : {image_filename: [<top-3 caption strings>]}
  t2i : {caption        : [<top-3 image_filename strings>]}
"""

import os
import json
import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from clip_model import CLIP
from dataset    import get_clip_val_transform
from tokenizer  import CLEVRTokenizer


# ──────────────────────────── Dataset ───────────────────────────────────── #

class CaptionImageDataset(Dataset):
    """
    Reads images using the absolute `image_path` field provided by the
    auto-grader. Returns the transformed image and the original index so we
    can recover the row order after batching.
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
        return self.transform(img), idx


# ──────────────────────────── Helpers ───────────────────────────────────── #

def _strip_compile_prefix(sd: dict) -> dict:
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def _patch_legacy_clip(sd: dict) -> dict:
    """Older CLIP ckpts have no `gap_proj.*`; mirror `image_proj` so the
    strict state-dict load succeeds. Retrieval uses `encode_image`, which
    consumes `image_proj` only, so this is a no-op semantically."""
    if 'gap_proj.weight' not in sd and 'image_proj.weight' in sd:
        sd['gap_proj.weight'] = sd['image_proj.weight'].clone()
        sd['gap_proj.bias']   = sd['image_proj.bias'].clone()
    return sd


def _load_clip(model_dir: str, device) -> CLIP:
    tok = CLEVRTokenizer.load(os.path.join(model_dir, 'tokenizer.json'))
    model = CLIP(vocab_size=tok.vocab_size).to(device)
    ckpt = torch.load(os.path.join(model_dir, 'clip.pt'),
                      map_location=device, weights_only=False)
    sd   = _patch_legacy_clip(_strip_compile_prefix(ckpt.get('model', ckpt)))
    model.load_state_dict(sd)
    model.eval()
    return model, tok


# ──────────────────────────── Main ──────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type',     required=True,
                   choices=['clip'])
    p.add_argument('--model_dir',      required=True)
    p.add_argument('--retrieval_task', required=True, choices=['i2t', 't2i'])
    p.add_argument('--data_path',      required=True)
    p.add_argument('--output_file',    required=True)
    p.add_argument('--batch_size',     type=int, default=256)
    p.add_argument('--num_workers',    type=int, default=8)
    p.add_argument('--max_seq_len',    type=int, default=77)
    p.add_argument('--top_k',          type=int, default=3,
                   help='Number of items to return per query (autograder = 3)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Read input JSON ─────────────────────────────────────────────────── #
    with open(args.data_path) as f:
        examples = json.load(f)

    captions  = [ex['caption']        for ex in examples]
    filenames = [ex['image_filename'] for ex in examples]

    # ── Load CLIP ──────────────────────────────────────────────────────── #
    model, tokenizer = _load_clip(args.model_dir, device)

    # ── Encode all images ──────────────────────────────────────────────── #
    transform = get_clip_val_transform()
    ds        = CaptionImageDataset(examples, transform)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    img_feats = [None] * len(examples)
    with torch.no_grad():
        for imgs, idx in loader:
            imgs = imgs.to(device, non_blocking=True)
            f    = model.encode_image(imgs).cpu()
            for j, b in zip(idx.tolist(), f):
                img_feats[j] = b

    # Stack to (N, 512); already L2-normalised by encode_image
    img_feats = torch.stack(img_feats, dim=0)

    # ── Encode all captions in parallel batches ────────────────────────── #
    txt_feats = []
    with torch.no_grad():
        for i in range(0, len(captions), args.batch_size):
            batch = captions[i : i + args.batch_size]
            ids_b, mask_b = [], []
            for c in batch:
                ids, mask = tokenizer.encode(c, max_len=args.max_seq_len)
                ids_b.append(torch.tensor(ids,  dtype=torch.long))
                mask_b.append(torch.tensor(mask, dtype=torch.long))
            ids_b  = torch.stack(ids_b,  dim=0).to(device)
            mask_b = torch.stack(mask_b, dim=0).to(device)
            txt_feats.append(model.encode_text(ids_b, mask_b).cpu())
    txt_feats = torch.cat(txt_feats, dim=0)

    # ── Cosine similarity matrix (already L2-normalised) ──────────────── #
    sim = img_feats @ txt_feats.T   # (N_img, N_txt)

    k_eff = min(args.top_k, sim.shape[1], sim.shape[0])

    output = {}
    if args.retrieval_task == 'i2t':
        # For each image, top-k captions
        topk_idx = sim.topk(k_eff, dim=1).indices.tolist()
        for img_idx, idxs in enumerate(topk_idx):
            output[filenames[img_idx]] = [captions[j] for j in idxs]
    else:  # t2i
        sim_t2i = sim.T  # (N_txt, N_img)
        topk_idx = sim_t2i.topk(k_eff, dim=1).indices.tolist()
        # The guidelines say we don't need to deduplicate captions on t2i,
        # but using a unique caption as a key is unambiguous; we keep first-
        # occurrence ranking which is well-defined for duplicate captions.
        seen = {}
        for txt_idx, idxs in enumerate(topk_idx):
            cap = captions[txt_idx]
            if cap in seen:
                continue
            seen[cap] = [filenames[j] for j in idxs]
        output = seen

    # ── Write ──────────────────────────────────────────────────────────── #
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Wrote {len(output)} entries to {args.output_file}')


if __name__ == '__main__':
    main()
