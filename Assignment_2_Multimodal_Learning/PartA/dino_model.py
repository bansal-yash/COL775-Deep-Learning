"""
dino_model.py - DINO self-supervised learning implementation.

Architecture (from assignment):
  Backbone  : same ViT as CLIP (12L, 6H, 384D, patch=16)
  Proj head : 3-layer MLP → 4096-dim output with L2-norm + weight-norm last layer
  Teacher   : EMA of student, receives only global crops
  Student   : receives all crops (2 global + 8 local)

Key components:
  - Multi-crop augmentation: 2×224 global + 8×96 local
  - Cross-entropy between teacher soft distribution and student log-softmax
  - Centering: running EMA of teacher outputs subtracted before softmax
  - Teacher temperature warmup from 0.04 to target
  - EMA momentum cosine schedule 0.996 → 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import VisionTransformer


# ──────────────────────────── Projection Head ────────────────────────────── #

class DINOHead(nn.Module):
    """
    DINO projection head: 3-layer MLP.

      Linear(in_dim, hidden_dim) → GELU
      Linear(hidden_dim, hidden_dim) → GELU
      Linear(hidden_dim, bottleneck_dim)
      L2-normalise
      WeightNorm Linear(bottleneck_dim, out_dim, bias=False)   ← last layer

    Assignment: out_dim = 4096.
    """
    def __init__(
        self,
        in_dim:        int = 384,
        hidden_dim:    int = 2048,
        bottleneck_dim:int = 256,
        out_dim:       int = 4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # Weight-normalised last layer with no bias (as in the DINO paper)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.utils.parametrizations.weight_norm(self.last_layer, name='weight')
        # Freeze magnitude (g=original0) — only direction (v=original1) is learned
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, out_dim)"""
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)   # L2-norm before last layer
        x = self.last_layer(x)
        return x


# ──────────────────────── Student / Teacher Network ──────────────────────── #

class DINONetwork(nn.Module):
    """Single DINO network (used for both student and teacher)."""
    def __init__(self, embed_dim: int = 384, out_dim: int = 4096,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=224, patch_size=16, in_channels=3,
            embed_dim=embed_dim, depth=12, num_heads=6, mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
        )
        self.head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            out_dim=out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, out_dim)"""
        cls = self.backbone(x)[:, 0]   # [CLS] token
        return self.head(cls)


# ──────────────────────────── DINO Framework ─────────────────────────────── #

class DINO(nn.Module):
    """
    Full DINO training framework.

    Usage
    ─────
      model = DINO()
      # views = [global1, global2, local1, ..., local8]  — list of (B, C, H, W)
      loss = model(views, epoch=current_epoch)
      loss.backward()
      # after optimiser step:
      model.update_teacher(momentum)
    """
    def __init__(
        self,
        embed_dim:              int   = 384,
        out_dim:                int   = 4096,
        student_temp:           float = 0.1,
        teacher_temp:           float = 0.07,
        teacher_temp_warmup_ep: int   = 30,
        teacher_temp_start:     float = 0.04,
        center_momentum:        float = 0.9,
        drop_path_rate:         float = 0.1,
    ):
        super().__init__()

        # Student uses stochastic depth; teacher does not (EMA copy, always eval)
        self.student = DINONetwork(embed_dim, out_dim, drop_path_rate=drop_path_rate)
        self.teacher = DINONetwork(embed_dim, out_dim, drop_path_rate=0.0)

        # Initialise teacher as a copy of student
        self.teacher.load_state_dict(self.student.state_dict())

        # Teacher never receives gradients
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student_temp           = student_temp
        self.teacher_temp           = teacher_temp
        self.teacher_temp_warmup_ep = teacher_temp_warmup_ep
        self.teacher_temp_start     = teacher_temp_start
        self.center_momentum        = center_momentum

        # Running center for teacher output centering (Eq. 4 in paper)
        self.register_buffer('center', torch.zeros(1, out_dim))

    # ─────────────────────────── EMA update ─────────────────────────────── #
    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Exponential moving average update of teacher parameters."""
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(momentum).add_((1.0 - momentum) * ps.data)

    # ─────────────────────────── Center update ──────────────────────────── #
    @torch.no_grad()
    def update_center(self, teacher_outputs: torch.Tensor):
        """
        Update running center (EMA of batch-mean teacher output).
        teacher_outputs: (B, D)
        """
        batch_mean = teacher_outputs.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            (1.0 - self.center_momentum) * batch_mean
        )

    # ─────────────────────────── Temperature schedule ───────────────────── #
    def get_teacher_temp(self, epoch: int) -> float:
        """Linear warmup of teacher temperature."""
        if self.teacher_temp_warmup_ep <= 0:
            return self.teacher_temp
        ratio = min(epoch / self.teacher_temp_warmup_ep, 1.0)
        return self.teacher_temp_start + ratio * (self.teacher_temp - self.teacher_temp_start)

    # ─────────────────────────── Forward / Loss ─────────────────────────── #
    def forward(self, views: list, epoch: int = 0) -> torch.Tensor:
        """
        Compute DINO loss for a batch of multi-crop views.

        views : list of tensors, each (B, C, H, W).
                views[0], views[1] are global crops (224x224).
                views[2:] are local crops (96x96).

        Returns: scalar loss.
        """
        n_global = 2   # always 2 global crops

        # ── Student forward: all views ──────────────────────────────────── #
        student_outs = [self.student(v) for v in views]   # list of (B, D)

        # ── Teacher forward: global crops only, no grad ─────────────────── #
        with torch.no_grad():
            t_raw = [self.teacher(views[i]) for i in range(n_global)]

            t_temp = self.get_teacher_temp(epoch)
            teacher_probs = []
            for t in t_raw:
                t_centered = t - self.center               # centering
                teacher_probs.append(
                    F.softmax(t_centered / t_temp, dim=-1)
                )

            # Update center with mean of teacher global outputs
            all_teacher = torch.stack(t_raw).mean(dim=0)  # (B, D)
            self.update_center(all_teacher)

        # ── DINO cross-entropy loss (Eq. 3 in paper) ────────────────────── #
        total_loss   = 0.0
        n_loss_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_out in enumerate(student_outs):
                if s_idx == t_idx:
                    continue    # skip same-view pair

                s_log = F.log_softmax(s_out / self.student_temp, dim=-1)
                loss  = -(t_prob * s_log).sum(dim=-1).mean()
                total_loss   += loss
                n_loss_terms += 1

        return total_loss / n_loss_terms

    # ─────────────────────────── Convenience accessors ──────────────────── #
    @property
    def student_backbone(self) -> VisionTransformer:
        return self.student.backbone

    @property
    def teacher_backbone(self) -> VisionTransformer:
        return self.teacher.backbone
