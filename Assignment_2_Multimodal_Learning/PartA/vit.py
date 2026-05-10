"""
vit.py - Vision Transformer backbone shared by CLIP and DINO.

Specification (from assignment):
  - 12 transformer layers, 6 attention heads, 384 hidden, 1536 MLP
  - Patch size 16, input 224x224
  - Output: (B, 197, 384) — token 0 is [CLS]

Enhancements for representation quality:
  - DropPath (stochastic depth) — standard regularisation for ViT
    (used in DeiT, DINO, MAE). Linearly increases from 0 to drop_path_rate
    across the depth of the network.
  - F.scaled_dot_product_attention (Flash Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── DropPath ────────────────────────────────────── #

class DropPath(nn.Module):
    """
    Stochastic Depth (per-sample drop of entire residual branch).
    Linearly ramps from 0 at the first layer to `drop_prob` at the last.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # per-sample binary mask: (B, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x * mask / keep


# ──────────────────────────── Patch Embedding ────────────────────────────── #

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ──────────────────────────── Transformer Block ───────────────────────────── #

class TransformerBlock(nn.Module):
    """
    Pre-norm block with DropPath on both attention and MLP residuals.
    Uses F.scaled_dot_product_attention (Flash Attention).
    """
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, 3 * dim, bias=True)
        self.proj  = nn.Linear(dim, dim)

        self.norm2   = nn.LayerNorm(dim)
        self.fc1     = nn.Linear(dim, mlp_dim)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, is_causal=False, attn_mask=None):
        B, N, C = x.shape

        # ── Self-attention ─────────────────────────────────────────────── #
        x_n = self.norm1(x)
        qkv = self.qkv(x_n).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.drop_path(self.proj(attn_out))

        # ── MLP ────────────────────────────────────────────────────────── #
        x_n = self.norm2(x)
        x = x + self.drop_path(self.dropout(self.fc2(self.act(self.fc1(x_n)))))
        return x


# ──────────────────────────── Vision Transformer ─────────────────────────── #

class VisionTransformer(nn.Module):
    """
    ViT backbone.

    drop_path_rate: maximum stochastic depth rate. Linearly increases
        from 0 at layer 0 to drop_path_rate at the last layer.
        Default 0.1 is the DeiT/DINO standard for ViT-Small.
    """
    def __init__(
        self,
        img_size:       int   = 224,
        patch_size:     int   = 16,
        in_channels:    int   = 3,
        embed_dim:      int   = 384,
        depth:          int   = 12,
        num_heads:      int   = 6,
        mlp_ratio:      float = 4.0,
        dropout:        float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        mlp_dim = int(embed_dim * mlp_ratio)
        # Linearly increasing drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def interpolate_pos_embed(self, num_patches):
        N_stored = self.pos_embed.shape[1] - 1
        if num_patches == N_stored:
            return self.pos_embed

        cls_pe   = self.pos_embed[:, :1, :]
        patch_pe = self.pos_embed[:, 1:, :]
        gs_old = int(N_stored ** 0.5)
        gs_new = int(num_patches ** 0.5)
        patch_pe = patch_pe.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe.float(), size=(gs_new, gs_new),
            mode='bicubic', align_corners=False,
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        return torch.cat([cls_pe, patch_pe], dim=1).to(self.pos_embed.dtype)

    def forward(self, x):
        B  = x.shape[0]
        x  = self.patch_embed(x)
        N  = x.shape[1]
        x  = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x  = x + self.interpolate_pos_embed(N)
        x  = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def get_cls_token(self, x):
        return self.forward(x)[:, 0]

    def get_gap_embedding(self, x):
        return self.forward(x)[:, 1:].mean(dim=1)

    def get_cls_and_gap(self, x):
        out = self.forward(x)
        return out[:, 0], out[:, 1:].mean(dim=1)

    def forward_intermediate(self, x, n_last: int = 4):
        """
        Return outputs from the last `n_last` transformer blocks.
        Each output has the final LayerNorm applied so feature scales match.

        Used for multi-layer feature concatenation in linear probing.
        Returns: list of n_last tensors, each (B, N+1, D).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        N = x.shape[1]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_embed(N)
        x = self.pos_drop(x)

        outputs = []
        cutoff = len(self.blocks) - n_last
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= cutoff:
                outputs.append(self.norm(x))
        return outputs
