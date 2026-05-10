"""
utils.py - Training utilities: LR scheduler, EMA momentum schedule,
           checkpointing, and metric helpers.
"""

import os
import math
import torch
import numpy as np
from sklearn.metrics import f1_score


# ═══════════════════════════ LR Schedulers ════════════════════════════════ #

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps:  int,
    num_training_steps: int,
    min_lr_ratio:       float = 0.0,
):
    """
    Cosine decay schedule with linear warmup.

    lr scales linearly from 0 → base_lr during warmup, then follows cosine
    decay from base_lr → min_lr_ratio * base_lr.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def cosine_momentum_schedule(
    step:        int,
    total_steps: int,
    start:       float = 0.996,
    end:         float = 1.0,
) -> float:
    """
    Cosine schedule for DINO teacher EMA momentum.
    Increases from `start` to `end` over `total_steps`.
    """
    return end - (end - start) * (math.cos(math.pi * step / total_steps) + 1) / 2


# ═══════════════════════════ Checkpointing ════════════════════════════════ #

def save_checkpoint(state: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None,
                    scheduler=None, device='cpu'):
    """Load checkpoint; returns start_epoch and best_metric."""
    ckpt = torch.load(filepath, map_location=device)
    model_state = {
        k.replace('_orig_mod.', ''): v
        for k, v in ckpt['model'].items()
    }
    model.load_state_dict(model_state)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', None)


# ═══════════════════════════ Metrics ══════════════════════════════════════ #

def accuracy_at_k(output: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """
    Top-k accuracy for single-label classification.
    output: (N, C) logits or probabilities
    target: (N,) integer labels
    """
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item() * 100.0


def multilabel_f1(preds: np.ndarray, targets: np.ndarray,
                  threshold: float = 0.5, average: str = 'macro') -> float:
    """
    F1 score for multi-label classification.
    preds:   (N, C) predicted probabilities
    targets: (N, C) ground-truth multi-hot labels
    """
    binary_preds = (preds >= threshold).astype(int)
    return f1_score(targets, binary_preds, average=average, zero_division=0) * 100.0


def recall_at_k(sim_matrix: np.ndarray, k: int) -> float:
    """
    Recall@K for retrieval.

    sim_matrix[i, j] = similarity between query i and gallery item j.
    Assumes diagonal is the ground-truth match (query i ↔ gallery i).

    Returns the percentage of queries for which the correct item
    appears in the top-K results.
    """
    N = sim_matrix.shape[0]
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]   # descending
    hits = sum(i in top_k_indices[i] for i in range(N))
    return hits / N * 100.0


# ═══════════════════════════ Misc ═════════════════════════════════════════ #

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device,
    use_cls:      bool = True,
    use_gap:      bool = False,
    is_clip_image:bool = False,
) -> torch.Tensor:
    """
    Extract frozen features from a vision encoder.

    Args:
        model:         ViT backbone (or CLIP model for image encoding)
        dataloader:    yields (images, ...) — only images[:] used
        use_cls:       return [CLS] token
        use_gap:       return global-average-pooled patch tokens
        is_clip_image: if True, calls model.encode_image() (returns normalised embeds)

    Returns:
        feats: (N, D) on CPU
    """
    model.eval()
    all_feats = []

    for batch in dataloader:
        images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        if is_clip_image:
            feats = model.encode_image(images)   # (B, 512)
        elif use_cls and use_gap:
            cls, gap = model.get_cls_and_gap(images)
            feats = torch.cat([cls, gap], dim=-1)
        elif use_gap:
            feats = model.get_gap_embedding(images)
        else:
            feats = model.get_cls_token(images)

        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)
