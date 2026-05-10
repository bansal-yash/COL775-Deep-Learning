"""
clip_model.py - CLIP model implementation.

Architecture (from assignment):
  Image encoder : ViT (12L, 6H, 384D, mlp=1536) → Linear(384,512) → L2-norm
  Text encoder  : Transformer (6L, 6H, 384D, mlp=1536) → Linear(384,512) → L2-norm
  Temperature   : learnable, initialised to log(1/0.07) (standard CLIP)
  Loss          : symmetric InfoNCE + auxiliary GAP contrastive loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import VisionTransformer, TransformerBlock


# ──────────────────────────── Text Encoder ───────────────────────────────── #

class TextEncoder(nn.Module):
    """
    Causal transformer text encoder for CLIP.

    Spec: 6 layers, 6 heads, 384 hidden dim, 1536 MLP dim.
    The representation of the final non-padding token (EOS) is projected to
    the 512-dimensional shared embedding space.
    """
    def __init__(
        self,
        vocab_size:  int,
        max_seq_len: int   = 77,
        embed_dim:   int   = 384,
        depth:       int   = 6,
        num_heads:   int   = 6,
        mlp_ratio:   float = 4.0,
        output_dim:  int   = 512,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(max_seq_len, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, L = tokens.shape
        device = tokens.device

        positions = torch.arange(L, device=device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, is_causal=True)

        x = self.norm(x)

        if attention_mask is not None:
            eos_idx = attention_mask.sum(dim=1).long() - 1
        else:
            eos_idx = torch.full((B,), L - 1, dtype=torch.long, device=device)

        batch_idx = torch.arange(B, device=device)
        x = x[batch_idx, eos_idx]

        return self.proj(x)


# ──────────────────────────── CLIP Model ─────────────────────────────────── #

class CLIP(nn.Module):
    """
    Standard CLIP with auxiliary GAP contrastive training.

      Image encoder: ViT-S/16 → Linear(384,512) → L2-norm   [CLS path, primary]
                              → Linear(384,512) → L2-norm   [GAP path, auxiliary]
      Text encoder : 6-layer causal transformer → Linear(384,512) → L2-norm
      Temperature  : learnable, init=log(1/0.07) (standard CLIP)

    Training loss = InfoNCE(CLS) + 0.5 * InfoNCE(GAP)
    The auxiliary GAP loss trains patch tokens to align with text, which
    directly improves both GAP probing accuracy and CLS through the shared backbone.
    """
    def __init__(
        self,
        vocab_size:      int,
        max_seq_len:     int   = 77,
        embed_dim:       int   = 384,
        output_dim:      int   = 512,
        drop_path_rate:  float = 0.05,
    ):
        super().__init__()

        self.image_encoder = VisionTransformer(
            img_size=224, patch_size=16, in_channels=3,
            embed_dim=embed_dim, depth=12, num_heads=6, mlp_ratio=4.0,
            drop_path_rate=drop_path_rate,
        )
        self.image_proj = nn.Linear(embed_dim, output_dim)
        self.gap_proj   = nn.Linear(embed_dim, output_dim)

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            output_dim=output_dim,
        )

        # Standard CLIP temperature: init to log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    # ─────────────────────────── encoders ─────────────────────────────────── #

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalised CLS embedding: (B, output_dim). Used at eval time."""
        tokens = self.image_encoder(images)
        return F.normalize(self.image_proj(tokens[:, 0]), dim=-1)

    def encode_text(self, tokens: torch.Tensor,
                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        """L2-normalised text embedding: (B, output_dim)."""
        return F.normalize(self.text_encoder(tokens, attention_mask), dim=-1)

    def _compute_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Scaled similarity matrix (B, B). Temperature clamped per CLIP paper."""
        t = self.logit_scale.exp().clamp(max=100.0)
        return t * (a @ b.T)

    # ─────────────────────────── forward ──────────────────────────────────── #

    def forward(self, images, tokens, attention_mask=None):
        """
        Returns:
          cls_logits : (B, B)  — CLS-based similarity (primary)
          gap_logits : (B, B)  — GAP-based similarity (auxiliary, for loss only)
          img_feat   : (B, D)  — L2-norm CLS image embedding
          txt_feat   : (B, D)  — L2-norm text embedding
        """
        vit_out  = self.image_encoder(images)          # (B, N+1, D)
        cls_emb  = F.normalize(self.image_proj(vit_out[:, 0]),          dim=-1)
        gap_emb  = F.normalize(self.gap_proj(vit_out[:, 1:].mean(1)),   dim=-1)
        txt_feat = self.encode_text(tokens, attention_mask)

        return (
            self._compute_logits(cls_emb, txt_feat),   # CLS logits
            self._compute_logits(gap_emb, txt_feat),   # GAP logits (auxiliary)
            cls_emb,
            txt_feat,
        )


# ──────────────────────────── Loss ───────────────────────────────────────── #

def clip_loss(logits_per_image: torch.Tensor,
              logits_per_text:  torch.Tensor) -> torch.Tensor:
    """
    Standard symmetric InfoNCE (CLIP) loss.

    Assumes each row i of logits_per_image has exactly one positive at column i
    (i.e. no duplicate captions in the batch). Use a full-dataset DataLoader
    with random shuffle — the probability of a duplicate landing in the same
    batch of 512 is negligible for CLEVR's caption distribution.
    """
    B      = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2.0
