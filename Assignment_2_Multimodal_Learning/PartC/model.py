import torch
import torch.nn as nn
import torch.nn.functional as F
import math

T_STEPS = 500
CFG_DROPOUT = 0.1
GUIDANCE_SCALE = 4.0
MODEL_CHANNELS = 256
NUM_HEADS = 8
CONTEXT_DIM = 512


# Residual block used in encoder and decoder
class residual_block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.GroupNorm(8, num_channels),
            nn.SiLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.GroupNorm(8, num_channels),
            nn.SiLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layer2(self.layer1(x))


# VAE architecture with encoder
class vae_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_first = nn.Conv2d(3, 32, 3, padding=1)

        self.stage1 = nn.Sequential(residual_block(32), residual_block(32))
        self.downsample1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.stage2 = nn.Sequential(residual_block(64), residual_block(64))
        self.downsample2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.stage3 = nn.Sequential(residual_block(128), residual_block(128))
        self.downsample3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.downsample1(self.stage1(x))
        x = self.downsample2(self.stage2(x))
        x = self.downsample3(self.stage3(x))

        return x


# Bottleneck block
class bottleneck_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.bottleneck = nn.Sequential(residual_block(128), residual_block(128))

    def forward(self, x):
        return self.bottleneck(x)


# Latent projection to get mean and logvar for reparameterization
class latent_projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_proj = nn.Conv2d(128, 8, 3, padding=1)

    def forward(self, x):
        return self.latent_proj(x)


# VAE decoder architecture
class vae_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(4, 128, 3, padding=1)

        self.mid = nn.Sequential(residual_block(128), residual_block(128))

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.stage1 = nn.Sequential(residual_block(128), residual_block(128))

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.stage2 = nn.Sequential(residual_block(64), residual_block(64))

        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.stage3 = nn.Sequential(residual_block(32), residual_block(32))

        self.final = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.conv_first(z)
        x = self.mid(x)

        x = self.stage1(self.conv1(self.upsample1(x)))
        x = self.stage2(self.conv2(self.upsample2(x)))
        x = self.stage3(self.conv3(self.upsample3(x)))

        return self.final(x)


# Full VAE model combining encoder, bottleneck, latent projection, and decoder
class vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vae_encoder()
        self.bottleneck = bottleneck_block()
        self.latent_proj = latent_projection()
        self.decoder = vae_decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h = self.bottleneck(h)
        h = self.latent_proj(h)

        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        return recon, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = self.bottleneck(h)
        h = self.latent_proj(h)

        mu, logvar = h.chunk(2, dim=1)

        return mu, logvar

    def decode(self, z):
        return self.decoder(z)


# Sinosidal embeddings of timesteps
class sin_timestep_embeds(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, t):
        half_dim = self.num_channels // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim) * -embeds).to(t.device)
        embeds = t[:, None] * embeds[None, :]
        embeds = torch.cat([torch.sin(embeds), torch.cos(embeds)], dim=-1)

        return embeds


# MLP Embeddings for timesteps
class mlp_timestep_embeds(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.time_dim = 4 * num_channels
        self.sinoidal_embeds = sin_timestep_embeds(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

    def forward(self, t):
        sinoidal_embeds = self.sinoidal_embeds(t)
        mlp_embeds = self.mlp(sinoidal_embeds)

        return mlp_embeds


# Residual block with time embeddings
class res_block_time(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):

        h = self.block1(x) + self.time_proj(t_emb)[:, :, None, None]
        return self.block2(h) + self.skip(x)


# Multi-head attention layer for LDM with optional cross-attention using context (e.g., text embeddings)
class multihead_attention_layer(nn.Module):
    def __init__(self, channels, num_heads, context_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        ctx_dim = context_dim if context_dim is not None else channels

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(ctx_dim, channels, bias=False)
        self.v_proj = nn.Linear(ctx_dim, channels, bias=False)

        self.to_out = nn.Linear(channels, channels)

    def forward(self, x, context=None):
        residual = x
        x = self.norm(x)
        ctx = context if context is not None else x
        B, N, C = x.shape
        Nc = ctx.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(ctx)
            .reshape(B, Nc, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(ctx)
            .reshape(B, Nc, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return residual + self.to_out(out)


# Spatial transformer block for LDM, combining self-attention and cross-attention with time embeddings
class spatial_transformer(nn.Module):
    def __init__(self, num_channels, num_heads, context_dim):
        super().__init__()

        self.gnorm = nn.GroupNorm(8, num_channels)
        self.proj_in = nn.Conv2d(num_channels, num_channels, 1)
        self.self_attn = multihead_attention_layer(num_channels, num_heads)
        self.cross_attn = multihead_attention_layer(
            num_channels, num_heads, context_dim
        )

        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, 4 * num_channels),
            nn.GELU(),
            nn.Linear(4 * num_channels, num_channels),
        )
        self.proj_out = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        residual = x
        x = self.proj_in(self.gnorm(x))
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        x = x + self.mlp(self.layer_norm(x))

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj_out(x) + residual


# Text conditioned UNet architecture for LDM, incorporating time embeddings and spatial transformers for image generation conditioned on text captions
class conditional_unet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        model_channels=MODEL_CHANNELS,
        context_dim=CONTEXT_DIM,
        num_heads=NUM_HEADS,
    ):
        super().__init__()
        ch, ch2 = model_channels, model_channels * 2
        tdim = model_channels * 4

        self.null_embedding = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)
        self.time_embed = mlp_timestep_embeds(model_channels)
        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.down_res_0 = nn.ModuleList(
            [res_block_time(ch, ch, tdim), res_block_time(ch, ch, tdim)]
        )
        self.down_attn_0 = spatial_transformer(ch, num_heads, context_dim)
        self.down_conv_0 = nn.Conv2d(ch, ch2, 3, stride=2, padding=1)

        self.down_res_1 = nn.ModuleList(
            [res_block_time(ch2, ch2, tdim), res_block_time(ch2, ch2, tdim)]
        )
        self.down_attn_1 = spatial_transformer(ch2, num_heads, context_dim)
        self.down_conv_1 = nn.Conv2d(ch2, ch2, 3, stride=2, padding=1)

        self.down_res_2 = nn.ModuleList(
            [res_block_time(ch2, ch2, tdim), res_block_time(ch2, ch2, tdim)]
        )
        self.down_attn_2 = spatial_transformer(ch2, num_heads, context_dim)

        self.mid_res1 = res_block_time(ch2, ch2, tdim)
        self.mid_attn = spatial_transformer(ch2, num_heads, context_dim)
        self.mid_res2 = res_block_time(ch2, ch2, tdim)

        self.up_res_2 = nn.ModuleList(
            [res_block_time(ch2 + ch2, ch2, tdim), res_block_time(ch2, ch2, tdim)]
        )
        self.up_attn_2 = spatial_transformer(ch2, num_heads, context_dim)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv_2 = nn.Conv2d(ch2, ch2, 3, padding=1)

        self.up_res_1 = nn.ModuleList(
            [res_block_time(ch2 + ch2, ch2, tdim), res_block_time(ch2, ch2, tdim)]
        )
        self.up_attn_1 = spatial_transformer(ch2, num_heads, context_dim)
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv_1 = nn.Conv2d(ch2, ch, 3, padding=1)

        self.up_res_0 = nn.ModuleList(
            [res_block_time(ch + ch, ch, tdim), res_block_time(ch, ch, tdim)]
        )
        self.up_attn_0 = spatial_transformer(ch, num_heads, context_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1)
        )

    def forward(self, x, t, context):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        for block in self.down_res_0:
            h = block(h, t_emb)
        h = self.down_attn_0(h, context)
        skip_0 = h
        h = self.down_conv_0(h)

        for block in self.down_res_1:
            h = block(h, t_emb)
        h = self.down_attn_1(h, context)
        skip_1 = h
        h = self.down_conv_1(h)

        for block in self.down_res_2:
            h = block(h, t_emb)
        h = self.down_attn_2(h, context)
        skip_2 = h

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        h = torch.cat([h, skip_2], dim=1)
        for block in self.up_res_2:
            h = block(h, t_emb)
        h = self.up_attn_2(h, context)
        h = self.up_conv_2(self.up_sample_2(h))

        h = torch.cat([h, skip_1], dim=1)
        for block in self.up_res_1:
            h = block(h, t_emb)
        h = self.up_attn_1(h, context)
        h = self.up_conv_1(self.up_sample_1(h))

        h = torch.cat([h, skip_0], dim=1)
        for block in self.up_res_0:
            h = block(h, t_emb)
        h = self.up_attn_0(h, context)

        return self.out(h)


# Cosine noise scheduling for diffusion
def cosine_betas(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-5, max=0.999).float()


# Diffusion scheduler
class diffusion_schedule:
    def __init__(self, T, device):
        self.T = T
        betas = cosine_betas(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * z0 + s * noise, noise
