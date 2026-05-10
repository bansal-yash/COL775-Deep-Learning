import json
import os
import pickle
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_image_id_set(path):
    if not path:
        return None
    with open(path, "rb") as f:
        ids = pickle.load(f)
    return set(ids)


def build_image_transform(image_size=224, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _image_allowed(item, ids):
    return ids is None or item.get("image_filename") in ids


def _limit_items(items, max_samples, shuffle, seed):
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    if max_samples:
        items = items[: int(max_samples)]
    return items


def iter_json_array(path, key):
    decoder = json.JSONDecoder()
    chunk_size = 1 << 20
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk
            key_pos = buf.find(f'"{key}"')
            if key_pos >= 0:
                arr_pos = buf.find("[", key_pos)
                if arr_pos >= 0:
                    pos = arr_pos + 1
                    break
            if len(buf) > chunk_size:
                buf = buf[-chunk_size:]
        while True:
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                break
            if pos >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf = buf[pos:] + chunk
                pos = 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
                yield obj
                pos = end
                if pos > chunk_size:
                    buf = buf[pos:]
                    pos = 0
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    raise
                buf = buf[pos:] + chunk
                pos = 0


class CaptionDataset(Dataset):
    def __init__(
        self,
        caption_path,
        image_root=None,
        transform=None,
        image_ids_path=None,
        max_samples=None,
        shuffle=False,
        seed=0,
    ):
        with open(caption_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        ids = load_image_id_set(image_ids_path)
        items = [x for x in items if _image_allowed(x, ids)]
        self.items = _limit_items(items, max_samples, shuffle, seed)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _load_image(self, filename):
        path = resolve_image_path(self.image_root, filename)
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        item = self.items[idx]
        caption = item["caption"]
        out = {
            "image_filename": item["image_filename"],
            "image_index": item["image_index"],
            "caption": caption,
        }
        if self.image_root:
            out["pixel_values"] = self._load_image(item["image_filename"])
        return out


class QADataset(Dataset):
    def __init__(
        self,
        qa_path,
        image_root=None,
        transform=None,
        image_ids_path=None,
        max_samples=None,
        shuffle=False,
        seed=0,
    ):
        ids = load_image_id_set(image_ids_path)
        items = []
        stop_early = max_samples and not shuffle
        for item in iter_json_array(qa_path, "questions"):
            if _image_allowed(item, ids):
                items.append(item)
                if stop_early and len(items) >= int(max_samples):
                    break
        self.items = _limit_items(items, max_samples, shuffle, seed)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _load_image(self, filename):
        path = resolve_image_path(self.image_root, filename)
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def _explanation(self, item, idx):
        facts = item.get("factual_explanation") or []
        counters = item.get("counter_factual_explanation") or []
        pool = facts or counters
        # idx-based selection: reproducible across workers, varies per sample
        return pool[idx % len(pool)] if pool else ""

    def __getitem__(self, idx):
        item = self.items[idx]
        answer = str(item["answer"])
        explanation = self._explanation(item, idx)
        target = f"Reasoning: {explanation}\nAnswer: {answer}"
        out = {
            "image_filename": item["image_filename"],
            "image_index": item["image_index"],
            "question": item["question"],
            "answer": answer,
            "target": target,
        }
        if self.image_root:
            out["pixel_values"] = self._load_image(item["image_filename"])
        return out


def vlm_collate(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        values = [x[key] for x in batch]
        if key == "pixel_values":
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def resolve_image_path(root, filename):
    root = Path(os.path.expanduser(os.path.expandvars(str(root))))
    split = "train" if "_train_" in filename else "val" if "_val_" in filename else ""
    candidates = [root / filename]
    if split:
        candidates.extend(
            [
                root / split / filename,
                root / "images" / split / filename,
                root / "Part_A" / split / filename,
                root / "Part_A" / split / "images" / filename,
                root / "Part_A" / "images" / split / filename,
                root / "Part_Aa" / split / filename,
                root / "Part_Aa" / split / "images" / filename,
                root / "Part_Aa" / "images" / split / filename,
            ]
        )
    candidates.extend(
        [
            root / "images" / filename,
            root / "Part_A" / filename,
            root / "Part_A" / "images" / filename,
            root / "Part_Aa" / filename,
            root / "Part_Aa" / "images" / filename,
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
