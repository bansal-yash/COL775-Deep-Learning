"""
dataset.py - Dataset classes and transforms for CLIP, DINO and linear probing.

Directory layout expected (relative to --data_root):
  Part_A/
    train/images/         (CLIP + DINO training images)
    train/clevr_train_captions.json
    val/images/
    val/clevr_val_captions.json
  Part_Aa/
    Clevr_official/images/  (probe images)
    Probe-Datasets/
      clevr_count_train.json  clevr_count_val.json
      clevr_colors_train.json clevr_colors_val.json
"""

import os
import json
import math
import random
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as T


# ═══════════════════════════════ Transforms ═══════════════════════════════ #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_BICUBIC = T.InterpolationMode.BICUBIC


def get_clip_train_transform() -> T.Compose:
    """
    CLEVR-tuned CLIP augmentation.

    Captions describe the whole scene, including object count and color set, so
    spatial crops that remove objects create label noise for contrastive
    training. We therefore keep the full scene and use only semantic-preserving
    augmentations:
      - deterministic resize + center crop to match evaluation geometry
      - horizontal flip (safe for CLEVR)
      - very light photometric jitter / blur for regularisation
    """
    return T.Compose([
        T.Resize(256, interpolation=_BICUBIC),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.12, 0.12, 0.06, 0.02)], p=0.35),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.2))], p=0.08),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_clip_val_transform() -> T.Compose:
    """Deterministic transform for CLIP validation / feature extraction."""
    return T.Compose([
        T.Resize(256, interpolation=_BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_clip_probe_transform(size: int = 256) -> T.Compose:
    """
    Higher-resolution eval transform for linear probing.
    Trained at 224 → evaluated at `size` (256/288). The ViT pos_embed
    is bicubic-interpolated to the new patch grid (16×16 at 256, 18×18 at 288),
    giving more spatial resolution for counting without changing parameters.
    """
    return T.Compose([
        T.Resize(int(round(size * 256 / 224)), interpolation=_BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiCropTransform:
    """
    DINO multi-crop augmentation (Section 2 in the DINO paper).

    Produces a list of 10 tensors per image:
      views[0], views[1]  — global crops  (224×224, scale ∈ [0.4, 1.0])
      views[2:]           — n_local local crops (96×96, scale ∈ [0.05, 0.4])
    """
    def __init__(self, n_local: int = 8, global_size: int = 224,
                 local_size: int = 96):
        norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        # No hue shift — CLEVR colours are strict semantic labels.
        # Any hue jitter teaches the model to ignore colour → destroys Color F1.
        cj = T.ColorJitter(brightness=0.3, contrast=0.3,
                            saturation=0.15, hue=0.0)

        self.global1 = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=_BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomApply([cj], p=0.8),
            # NO grayscale — colours are semantic in CLEVR
            T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=1.0),
            T.ToTensor(), norm,
        ])
        self.global2 = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=_BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomApply([cj], p=0.8),
            T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.1),
            T.RandomSolarize(threshold=128, p=0.2),
            T.ToTensor(), norm,
        ])
        self.local_t = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4), interpolation=_BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomApply([cj], p=0.8),
            T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(), norm,
        ])
        self.n_local = n_local

    def __call__(self, img) -> List[torch.Tensor]:
        crops  = [self.global1(img), self.global2(img)]
        crops += [self.local_t(img) for _ in range(self.n_local)]
        return crops   # list of 10 tensors


# ═══════════════════════════════ CLIP Dataset ════════════════════════════ #

class CLIPDataset(Dataset):
    """
    (image, caption) pair dataset for CLIP training and validation.

    Supports multiple JSON structures:
      list of {"image": ..., "caption": ...}
      dict   {filename: caption_string}
      dict   {filename: {"caption": ...}}
    """
    def __init__(
        self,
        image_dir:    str,
        captions_file:str,
        tokenizer,
        max_seq_len:  int  = 77,
        transform     = None,
        is_train:     bool = True,
    ):
        self.image_dir   = image_dir
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.transform   = (transform
                            or (get_clip_train_transform() if is_train
                                else get_clip_val_transform()))

        with open(captions_file) as f:
            raw = json.load(f)
        self.samples: List[Tuple[str, str]] = self._parse(raw)

    # ─────────────────────────── parsing ─────────────────────────────────── #
    def _parse(self, raw) -> List[Tuple[str, str]]:
        samples = []
        _img_keys = ('image', 'filename', 'file_name', 'image_filename')
        _cap_keys = ('caption', 'text', 'description')

        if isinstance(raw, list):
            for item in raw:
                img = next((item[k] for k in _img_keys if k in item), None)
                cap = next((item[k] for k in _cap_keys if k in item), None)
                if img and cap:
                    samples.append((img, cap))
        elif isinstance(raw, dict):
            for img, v in raw.items():
                if isinstance(v, str):
                    samples.append((img, v))
                elif isinstance(v, dict):
                    cap = next((v[k] for k in _cap_keys if k in v), None)
                    if cap:
                        samples.append((img, cap))
        return samples

    def get_all_captions(self) -> List[str]:
        return [cap for _, cap in self.samples]

    # ─────────────────────────── dataset interface ──────────────────────── #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        img = self.transform(img)

        ids, mask = self.tokenizer.encode(caption, max_len=self.max_seq_len)
        tokens    = torch.tensor(ids,  dtype=torch.long)
        attn_mask = torch.tensor(mask, dtype=torch.long)

        return img, tokens, attn_mask, caption


# ═════════════════════════ Unique-Caption Sampler ════════════════════════ #

class UniqueCaptionBatchSampler(Sampler):
    """
    Constructs caption-diverse batches while still using the full dataset.

    In CLEVR many images share identical captions (same scene description).
    Including duplicates in a contrastive batch creates ambiguous negatives.
    This sampler greedily fills each batch with at most one sample per caption
    for as long as that is possible. Only when the number of active captions
    falls below the batch size do duplicate captions reappear within a batch.

    Unlike the earlier "one sample per caption per epoch" version, this uses
    all samples every epoch, so we keep data efficiency while still reducing
    duplicate-caption collisions.
    """
    def __init__(self, captions: List[str], batch_size: int,
                 drop_last: bool = True, shuffle: bool = True):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.total_samples = len(captions)

        cap2idx: dict = defaultdict(list)
        for i, c in enumerate(captions):
            cap2idx[c].append(i)
        self.groups: List[List[int]] = list(cap2idx.values())

    def __iter__(self):
        groups = [list(g) for g in self.groups]
        if self.shuffle:
            for g in groups:
                random.shuffle(g)

        remaining = sum(len(g) for g in groups)
        while remaining > 0:
            batch = []

            # First pass: maximise caption diversity (one sample per caption).
            active = [i for i, g in enumerate(groups) if g]
            if self.shuffle:
                random.shuffle(active)
            for gi in active:
                if len(batch) == self.batch_size:
                    break
                batch.append(groups[gi].pop())
                remaining -= 1

            # If we still need more samples, duplicates are unavoidable.
            while len(batch) < self.batch_size and remaining > 0:
                refill = [i for i, g in enumerate(groups) if g]
                if self.shuffle:
                    random.shuffle(refill)
                for gi in refill:
                    if len(batch) == self.batch_size:
                        break
                    if not groups[gi]:
                        continue
                    batch.append(groups[gi].pop())
                    remaining -= 1

            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)


# ═══════════════════════════════ DINO Dataset ════════════════════════════ #

class DINODataset(Dataset):
    """
    Image-only dataset for DINO training.
    Returns a list of 10 augmented views per image.
    """
    def __init__(self, image_dir: str, n_local: int = 8):
        self.image_dir = image_dir
        self.transform = MultiCropTransform(n_local=n_local)
        self.files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img  = Image.open(os.path.join(self.image_dir, self.files[idx])).convert('RGB')
        return self.transform(img)   # list of 10 tensors


def dino_collate_fn(batch: list) -> List[torch.Tensor]:
    """Stack each view index across the batch."""
    n_views = len(batch[0])
    return [torch.stack([b[v] for b in batch]) for v in range(n_views)]


# ═════════════════════════════ Probe Datasets ════════════════════════════ #

CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
COLOR2IDX    = {c: i for i, c in enumerate(CLEVR_COLORS)}
N_COLORS     = len(CLEVR_COLORS)


def _find(d: dict, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


class CLEVRCountDataset(Dataset):
    """
    Object-count probe dataset.
    Label: integer in range [min_count, max_count]; shifted to start at 0.
    """
    def __init__(self, image_dir: str, json_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform or get_clip_val_transform()

        with open(json_file) as f:
            raw = json.load(f)
        self.samples = self._parse(raw)

        counts = [c for _, c in self.samples]
        self.min_count = min(counts)
        self.max_count = max(counts)
        self.n_classes = self.max_count - self.min_count + 1

    @staticmethod
    def _extract_count(value, count_keys) -> int:
        """Recursively unwrap a count value that might be a dict, list, float, or int."""
        if isinstance(value, dict):
            for k in count_keys:
                if k in value:
                    return CLEVRCountDataset._extract_count(value[k], count_keys)
            # fallback: try the first numeric value found
            for v in value.values():
                if isinstance(v, (int, float)):
                    return int(v)
            return 0
        elif isinstance(value, list):
            return int(value[0]) if len(value) > 0 else 0
        else:
            return int(value)

    def _parse(self, raw):
        # Handle "examples" wrapper if present
        if isinstance(raw, dict) and 'examples' in raw:
            raw = raw['examples']

        samples = []
        _img_keys   = ('image', 'filename', 'file_name', 'image_filename')
        _count_keys = ('count', 'label', 'num_objects', 'n_objects', 'object_count')

        if isinstance(raw, list):
            for item in raw:
                img   = _find(item, *_img_keys)
                split = item.get('split', '')
                if split and not img.startswith(f"{split}/"):
                    # Check if the img already exists relative to image_dir; 
                    # if not, prepend the split if helpful.
                    img = os.path.join(split, img)
                
                count = _find(item, *_count_keys, default=0)
                samples.append((img, self._extract_count(count, _count_keys)))
        elif isinstance(raw, dict):
            for img, v in raw.items():
                samples.append((img, self._extract_count(
                    _find(v, *_count_keys, default=v) if isinstance(v, dict) else v,
                    _count_keys
                )))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, count = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        img = self.transform(img)
        label = count - self.min_count   # 0-indexed
        return img, label


class CLEVRColorDataset(Dataset):
    """
    Color-prediction probe dataset (multi-label).
    Label: float32 multi-hot vector of length 8 (one per CLEVR color).
    """
    def __init__(self, image_dir: str, json_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform or get_clip_val_transform()

        with open(json_file) as f:
            raw = json.load(f)
        self.samples = self._parse(raw)

    def _parse(self, raw):
        # Handle "examples" wrapper if present
        if isinstance(raw, dict) and 'examples' in raw:
            raw = raw['examples']

        samples = []
        _img_keys   = ('image', 'filename', 'file_name', 'image_filename')
        _color_keys = ('colors', 'label', 'color_list', 'color_names', 'active_labels')
        
        if isinstance(raw, list):
            for item in raw:
                img    = _find(item, *_img_keys)
                split  = item.get('split', '')
                if split and not img.startswith(f"{split}/"):
                    img = os.path.join(split, img)
                
                colors = _find(item, *_color_keys, default=[])
                if isinstance(colors, str):
                    colors = [colors]
                samples.append((img, colors))
        elif isinstance(raw, dict):
            for img, v in raw.items():
                if isinstance(v, dict):
                    colors = _find(v, *_color_keys, default=[])
                elif isinstance(v, list):
                    colors = v
                else:
                    colors = []
                samples.append((img, colors))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, colors = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        img = self.transform(img)

        label = torch.zeros(N_COLORS, dtype=torch.float32)
        for c in colors:
            c_lower = c.lower()
            if c_lower in COLOR2IDX:
                label[COLOR2IDX[c_lower]] = 1.0

        return img, label
