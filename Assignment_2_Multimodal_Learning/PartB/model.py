import importlib
import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils import get_dtype, set_trainable


DEFAULT_VISION_CHECKPOINT = "/scratch/cse/btech/cs1221089/clip-safe/best.pt"


def import_object(path):
    if ":" in path:
        module_name, name = path.split(":", 1)
    else:
        module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def pick_sub_state(state):
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    prefixes = (
        "vision_encoder.",
        "image_encoder.",
        "visual_encoder.",
        "vision_model.",
        "image_model.",
        "encoder.",
        "module.vision_encoder.",
        "module.image_encoder.",
        "module.visual_encoder.",
        "module.",
    )
    best_prefix, best_count = "", 0
    for prefix in prefixes:
        count = sum(1 for k in state if k.startswith(prefix))
        if count > best_count:
            best_prefix, best_count = prefix, count
    if best_prefix:
        return {k[len(best_prefix) :]: v for k, v in state.items() if k.startswith(best_prefix)}
    return state


def extract_state_dict(obj, key=None):
    if isinstance(obj, nn.Module):
        return None, obj
    if key and isinstance(obj, dict) and key in obj:
        obj = obj[key]
    if isinstance(obj, dict):
        for k in ("vision_encoder", "image_encoder", "visual_encoder", "state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return pick_sub_state(obj[k]), None
        return pick_sub_state(obj), None
    raise ValueError("Unsupported vision checkpoint format")


def strip_prefix(state, prefix):
    if not prefix:
        return state
    prefix = prefix.rstrip(".") + "."
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


def _remap_vit_keys(state):
    """
    Remap keys from vit.py VisionTransformer / TransformerBlock format
    to AssignmentVisionEncoder / ViTBlock format.

    Key differences:
      patch_embed.proj.*  →  patch_embed.*         (PatchEmbed wrapper vs bare Conv2d)
      blocks.X.qkv.*      →  blocks.X.attn.in_proj_*   (combined QKV linear vs MultiheadAttention)
      blocks.X.proj.*      →  blocks.X.attn.out_proj.*  (output projection)
      blocks.X.fc1.*       →  blocks.X.mlp.0.*          (MLP first linear)
      blocks.X.fc2.*       →  blocks.X.mlp.3.*          (MLP second linear, index 3 due to GELU+Dropout)
    """
    new_state = {}
    for k, v in state.items():
        new_k = k
        # PatchEmbed wrapper → bare Conv2d
        if k.startswith("patch_embed.proj."):
            new_k = k.replace("patch_embed.proj.", "patch_embed.")
        # Attention QKV
        elif ".qkv.weight" in k:
            new_k = k.replace(".qkv.weight", ".attn.in_proj_weight")
        elif ".qkv.bias" in k:
            new_k = k.replace(".qkv.bias", ".attn.in_proj_bias")
        # Attention output projection (only inside blocks, not top-level)
        elif "blocks." in k and ".proj.weight" in k:
            new_k = k.replace(".proj.weight", ".attn.out_proj.weight")
        elif "blocks." in k and ".proj.bias" in k:
            new_k = k.replace(".proj.bias", ".attn.out_proj.bias")
        # MLP fc1 → mlp.0, fc2 → mlp.3
        elif ".fc1.weight" in k:
            new_k = k.replace(".fc1.weight", ".mlp.0.weight")
        elif ".fc1.bias" in k:
            new_k = k.replace(".fc1.bias", ".mlp.0.bias")
        elif ".fc2.weight" in k:
            new_k = k.replace(".fc2.weight", ".mlp.3.weight")
        elif ".fc2.bias" in k:
            new_k = k.replace(".fc2.bias", ".mlp.3.bias")
        # Skip DropPath / pos_drop (no learnable params)
        elif "drop_path" in k or "pos_drop" in k:
            continue
        new_state[new_k] = v
    return new_state


def build_vision_encoder(config):
    model_name = config.get("vision_model_name")
    checkpoint = os.environ.get("A2_VISION_CHECKPOINT", config.get("vision_checkpoint") or DEFAULT_VISION_CHECKPOINT)
    class_path = config.get("vision_class")
    if model_name:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    elif class_path:
        cls = import_object(class_path)
        model = cls(**config.get("vision_kwargs", {}))
    elif checkpoint:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state, module = extract_state_dict(obj, config.get("vision_state_key"))
        if module is None:
            model = AssignmentVisionEncoder(**config.get("vision_kwargs", {}))
            state = strip_prefix(state, config.get("vision_strip_prefix"))
            matched = set(state) & set(model.state_dict())
            if not matched:
                # Try remapping from vit.py VisionTransformer key format
                state = _remap_vit_keys(state)
                matched = set(state) & set(model.state_dict())
            if not matched:
                ckpt_keys = sorted(state.keys())[:10]
                model_keys = sorted(model.state_dict().keys())[:10]
                raise ValueError(
                    f"No matching vision checkpoint keys found for AssignmentVisionEncoder.\n"
                    f"  Checkpoint keys (first 10): {ckpt_keys}\n"
                    f"  Model keys     (first 10): {model_keys}"
                )
            model.load_state_dict(state, strict=bool(config.get("vision_strict_load", False)))
        else:
            model = module
    else:
        raise ValueError("Set vision_checkpoint with vision_class or set vision_model_name")
    if checkpoint and class_path:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state, module = extract_state_dict(obj, config.get("vision_state_key"))
        if module is not None:
            model = module
        else:
            state = strip_prefix(state, config.get("vision_strip_prefix"))
            model.load_state_dict(state, strict=bool(config.get("vision_strict_load", True)))
    set_trainable(model, False)
    model.eval()
    return model


class ViTBlock(nn.Module):
    def __init__(self, dim=384, heads=6, mlp_dim=1536, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        normed = self.norm1(x)
        y, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class AssignmentVisionEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.0,
        **_,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed[:, : x.size(1)])
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ReverseBottleneckProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, activation="gelu", norm=True):
        super().__init__()
        acts = {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}
        layers = []
        if norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                acts.get(activation, nn.GELU()),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VLMForCausalLM(nn.Module):
    def __init__(self, vision_encoder, llm, tokenizer, projector, drop_cls_token=True, max_length=256):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.tokenizer = tokenizer
        self.projector = projector
        self.drop_cls_token = drop_cls_token
        self.max_length = max_length

    @property
    def embed_tokens(self):
        return self.llm.get_input_embeddings()

    def format_prompt(self, user_text):
        messages = [{"role": "user", "content": user_text}]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"User: {user_text}\nAssistant:"

    def _extract_tokens(self, output):
        if isinstance(output, dict):
            for key in ("patch_tokens", "last_hidden_state", "tokens"):
                if key in output:
                    tokens = output[key]
                    break
            else:
                raise ValueError("Vision encoder output does not contain patch tokens")
        elif hasattr(output, "last_hidden_state"):
            tokens = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            tokens = output[0]
        else:
            tokens = output
        if tokens.ndim != 3:
            raise ValueError("Vision encoder must return all patch tokens as B x N x D")
        if self.drop_cls_token and tokens.size(1) > 1:
            tokens = tokens[:, 1:]
        return tokens

    def encode_image(self, pixel_values=None, vision_tokens=None):
        if vision_tokens is None:
            if pixel_values is None:
                raise ValueError("pixel_values or vision_tokens required")
            with torch.no_grad():
                pixel_values = pixel_values.to(dtype=next(self.vision_encoder.parameters()).dtype)
                output = self.vision_encoder(pixel_values)
                vision_tokens = self._extract_tokens(output)
        vision_tokens = vision_tokens.to(dtype=next(self.projector.parameters()).dtype)
        return self.projector(vision_tokens)

    def _tokenize(self, texts, add_special_tokens=False):
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

    def _build_batch(self, image_embeds, prompts, targets=None):
        device = image_embeds.device
        prompt_ids = self._tokenize([self.format_prompt(x) for x in prompts])
        if targets is not None:
            eos = self.tokenizer.eos_token or ""
            target_ids = self._tokenize([x + eos for x in targets])
        else:
            target_ids = [[] for _ in prompts]
        rows, labels, input_ids = [], [], []
        for i, (pids, tids) in enumerate(zip(prompt_ids, target_ids)):
            ids = pids + tids
            text_ids = torch.tensor(ids, device=device, dtype=torch.long)
            text_embeds = self.embed_tokens(text_ids)
            img = image_embeds[i].to(dtype=text_embeds.dtype)
            row = torch.cat([img, text_embeds], dim=0)
            lab = torch.full((row.size(0),), -100, device=device, dtype=torch.long)
            if targets is not None and tids:
                lab[-len(tids) :] = torch.tensor(tids, device=device, dtype=torch.long)
            rows.append(row)
            labels.append(lab)
            input_ids.append(torch.tensor([self.tokenizer.pad_token_id] * image_embeds.size(1) + ids, device=device))
        max_len = max(x.size(0) for x in rows)
        hidden = rows[0].size(-1)
        embeds = torch.zeros(len(rows), max_len, hidden, device=device, dtype=rows[0].dtype)
        masks = torch.zeros(len(rows), max_len, device=device, dtype=torch.long)
        label_pad = torch.full((len(rows), max_len), -100, device=device, dtype=torch.long)
        id_pad = torch.full((len(rows), max_len), self.tokenizer.pad_token_id, device=device, dtype=torch.long)
        for i, row in enumerate(rows):
            n = row.size(0)
            embeds[i, :n] = row
            masks[i, :n] = 1
            label_pad[i, :n] = labels[i]
            id_pad[i, :n] = input_ids[i]
        return embeds, masks, label_pad, id_pad

    def forward(self, pixel_values=None, vision_tokens=None, prompts=None, targets=None):
        image_embeds = self.encode_image(pixel_values=pixel_values, vision_tokens=vision_tokens)
        inputs_embeds, attention_mask, labels, _ = self._build_batch(image_embeds, prompts, targets)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)

    @torch.no_grad()
    def generate_text(self, pixel_values=None, vision_tokens=None, prompts=None, max_new_tokens=96, use_cache=False):
        image_embeds = self.encode_image(pixel_values=pixel_values, vision_tokens=vision_tokens)
        inputs_embeds, attention_mask, _, input_ids = self._build_batch(image_embeds, prompts, None)
        output = self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=use_cache,
        )
        gen = output[:, input_ids.size(1) :]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def load_projector(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt["projector"] if isinstance(ckpt, dict) and "projector" in ckpt else ckpt
        self.projector.load_state_dict(state)


def resolve_llm_path(model_name):
    """Return local path from A2_LLM_PATH env var if set, otherwise the HF model ID."""
    return os.environ.get("A2_LLM_PATH") or model_name


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(resolve_llm_path(model_name), trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_llm(config, stage):
    dtype = get_dtype(config.get("precision", "bf16"))
    _kwargs = dict(trust_remote_code=True, attn_implementation=config.get("attn_implementation", "sdpa"))
    try:
        llm = AutoModelForCausalLM.from_pretrained(resolve_llm_path(config["llm_model_name"]), dtype=dtype, **_kwargs)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(resolve_llm_path(config["llm_model_name"]), torch_dtype=dtype, **_kwargs)
    llm.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        try:
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            llm.gradient_checkpointing_enable()
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
    set_trainable(llm, False)
    return llm


def apply_lora(llm, lora_cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    return get_peft_model(llm, cfg)


def build_vlm(config, stage):
    tokenizer = build_tokenizer(config["llm_model_name"])
    llm = build_llm(config, stage)
    if stage == "stage2":
        llm = apply_lora(llm, config["stage2"].get("lora", {}))
        # LoRA params must stay FP32 — GradScaler.unscale_ rejects FP16 grads,
        # and FP32 master weights are the standard mixed-precision pattern.
        for p in llm.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    vision = build_vision_encoder(config)
    if config.get("cast_frozen_vision", True):
        vision = vision.to(dtype=get_dtype(config.get("precision", "bf16")))
    llm_hidden = llm.config.hidden_size
    pcfg = config["projector"]
    # Projector kept in FP32; autocast handles compute-dtype casting in forward.
    projector = ReverseBottleneckProjector(
        input_dim=int(pcfg["input_dim"]),
        hidden_dim=int(pcfg["hidden_dim"]),
        output_dim=int(llm_hidden),
        dropout=float(pcfg.get("dropout", 0.0)),
        activation=pcfg.get("activation", "gelu"),
        norm=bool(pcfg.get("norm", True)),
    )
    model = VLMForCausalLM(
        vision_encoder=vision,
        llm=llm,
        tokenizer=tokenizer,
        projector=projector,
        drop_cls_token=bool(config.get("vision_drop_cls_token", True)),
        max_length=int(config.get(stage, {}).get("max_length", 256)),
    )
    stage_cfg = config.get(stage, {})
    if stage == "stage2" and stage_cfg.get("stage1_checkpoint"):
        model.load_projector(stage_cfg["stage1_checkpoint"])
    # Final safety: every trainable param must be FP32 (master weights for AMP).
    # GradScaler.unscale_ raises on FP16 grads; this catches any path that may
    # have left a trainable param in compute dtype.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model
