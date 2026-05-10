import argparse
import json
import os
from pathlib import Path

# Force offline mode so transformers/HF only read from the local cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import build_image_transform
from model import build_vlm
from utils import get_device, load_config, seed_everything


DEFAULT_PROMPT = "Question: {question}\nReason step by step, then give the final answer."


def find_artifact(model_dir, candidate_names):
    for name in candidate_names:
        cand = model_dir / name
        if cand.exists():
            return cand
    return None


def find_lora_dir(model_dir):
    for cand in [model_dir / "best_lora", model_dir / "lora", model_dir]:
        if cand.is_dir() and (cand / "adapter_config.json").exists():
            return cand
    for sub in model_dir.rglob("adapter_config.json"):
        return sub.parent
    return None


def parse_generation(text):
    """Split a generated string into (reasoning, answer) using the last 'answer:' marker."""
    text = str(text).strip()
    low = text.lower()
    marker = "answer:"
    if marker in low:
        i = low.rfind(marker)
        reasoning = text[:i].strip()
        tail = text[i + len(marker):].strip()
        answer = tail.splitlines()[0].strip(" .") if tail else ""
    else:
        reasoning = text
        answer = text.splitlines()[0].strip(" .") if text else ""
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return reasoning, answer


class QuestionDataset(Dataset):
    def __init__(self, questions, transform, image_root=None):
        self.items = questions
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.items)

    def _resolve(self, q):
        if q.get("image_path"):
            return q["image_path"]
        fn = q["image_filename"]
        if self.image_root:
            for sub in ("", "train", "val", "test"):
                cand = os.path.join(self.image_root, sub, fn) if sub else os.path.join(self.image_root, fn)
                if os.path.exists(cand):
                    return cand
        return fn

    def __getitem__(self, idx):
        q = self.items[idx]
        path = self._resolve(q)
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "question": q["question"],
            "question_index": q.get("question_index", idx),
        }


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "question": [b["question"] for b in batch],
        "question_index": [b["question_index"] for b in batch],
    }


def build_model(model_dir, config, device):
    # Resolve the bundled vision checkpoint
    vision_ckpt = find_artifact(
        model_dir, ["vision_encoder.pt", "vision.pt", "best.pt", "clip_best.pt"]
    )
    if vision_ckpt is not None:
        config["vision_checkpoint"] = str(vision_ckpt)
        os.environ["A2_VISION_CHECKPOINT"] = str(vision_ckpt)

    config.setdefault("stage2", {})["stage1_checkpoint"] = None
    config["gradient_checkpointing"] = False

    # 1) Build base VLM (no LoRA), move to device
    model = build_vlm(config, "stage1").to(device)

    # 2) Load trained projector weights
    projector_path = find_artifact(
        model_dir,
        [
            "best_stage2_projector.pt",
            "stage2_projector.pt",
            "projector.pt",
            "last_stage2_projector.pt",
        ],
    )
    if projector_path is None:
        raise FileNotFoundError(f"No projector .pt found under {model_dir}")
    model.load_projector(str(projector_path))

    # 3) Attach trained LoRA adapter from disk, then move newly loaded LoRA onto device
    lora_dir = find_lora_dir(model_dir)
    if lora_dir is None:
        raise FileNotFoundError(f"No LoRA adapter directory found under {model_dir}")
    model.llm = PeftModel.from_pretrained(model.llm, str(lora_dir))
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--image_root", type=str, default=None,
                        help="Optional. Local fallback when input JSON lacks 'image_path'.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    config_path = find_artifact(model_dir, ["config.json"])
    if config_path is None:
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    config = load_config(str(config_path))

    seed_everything(config.get("seed", 0))
    device = get_device()
    model = build_model(model_dir, config, device)

    with open(args.data_path, "r") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) else data

    transform = build_image_transform(
        image_size=config.get("image_size", 224),
        mean=config.get("image_mean"),
        std=config.get("image_std"),
    )
    loader = DataLoader(
        QuestionDataset(questions, transform, image_root=args.image_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    prompt_template = config.get("stage2", {}).get("prompt", DEFAULT_PROMPT)
    max_new_tokens = int(config.get("stage2", {}).get("max_new_tokens", args.max_new_tokens))

    print(f"PartB inference | samples={len(questions)} | batches={len(loader)} | bs={args.batch_size} | max_new_tokens={max_new_tokens}")

    results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="PartB inference"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            prompts = [prompt_template.format(question=q) for q in batch["question"]]
            texts = model.generate_text(
                pixel_values=pixel_values,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            for qi, txt in zip(batch["question_index"], texts):
                reasoning, answer = parse_generation(txt)
                results[str(qi)] = {"reasoning": reasoning, "answer": answer}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Wrote {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
