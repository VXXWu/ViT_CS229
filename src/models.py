"""Model definitions for ViT experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import DropPath


# =============================================================================
# Custom ViT with PyTorch Scaled Dot-Product Attention (SDPA)
# =============================================================================

class SDPAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention."""

    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # PyTorch native SDPA — uses FlashAttention or Memory-Efficient kernels on CUDA
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SDPViT(nn.Module):
    """Vision Transformer with PyTorch native scaled dot-product attention.

    ViT-Small config: embed_dim=384, depth=12, num_heads=6, patch_size=16.
    Uses F.scaled_dot_product_attention for fused FlashAttention/MemEfficient
    kernels on CUDA — ~2x faster and ~0.5x memory vs manual attention.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x[:, 0])  # CLS token

    def get_patch_tokens(self, x):
        """Extract patch tokens (excluding CLS) from the final layer."""
        features = self.forward_features(x)
        return features[:, 1:, :]

    def get_attention_maps(self, x):
        """Compute attention maps for each layer.

        Since SDPA doesn't return attention weights, we recompute Q*K^T
        manually from each block's qkv projection for analysis only.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attention_maps = []
        for block in self.blocks:
            # Recompute attention weights from pre-norm input
            normed = block.norm1(x)
            BN, N, C = normed.shape
            qkv = block.attn.qkv(normed).reshape(BN, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu())

            # Run the actual block forward
            x = block(x)

        return attention_maps


# =============================================================================
# Register ViT (SDPA + learnable register tokens)
# =============================================================================

class RegisterViT(nn.Module):
    """Vision Transformer with register tokens (Darcet et al., 2023).

    Identical to SDPViT but adds k learnable register tokens that participate
    in attention alongside CLS and patch tokens. Register tokens are discarded
    before the classification head. They absorb high-norm artifacts that would
    otherwise appear in patch tokens.

    Token layout: [CLS, register_1, ..., register_k, patch_1, ..., patch_196]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05, num_register_tokens=4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + register tokens + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        # Positional embedding covers CLS + patches only (registers have no position)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks (same as SDPViT)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # [CLS, patches]
        x = self.pos_drop(x + self.pos_embed)        # positional embed on CLS + patches

        # Insert register tokens after CLS (no positional embedding for registers)
        reg = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)  # [CLS, reg_1..k, patches]

        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x[:, 0])  # CLS token

    def get_patch_tokens(self, x):
        """Extract patch tokens (excluding CLS and register tokens)."""
        features = self.forward_features(x)
        return features[:, 1 + self.num_register_tokens:, :]

    def get_register_tokens(self, x):
        """Extract register tokens for analysis."""
        features = self.forward_features(x)
        return features[:, 1:1 + self.num_register_tokens, :]

    def get_attention_maps(self, x):
        """Compute attention maps for each layer (includes register tokens)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        reg = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        attention_maps = []
        for block in self.blocks:
            normed = block.norm1(x)
            BN, N, C = normed.shape
            qkv = block.attn.qkv(normed).reshape(BN, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu())

            x = block(x)

        return attention_maps


# =============================================================================
# Vanilla ViT (timm DeiT-III wrapper, kept for comparison)
# =============================================================================

class VanillaViT(nn.Module):
    """Thin wrapper around DeiT-III Small for ImageNet-100."""

    def __init__(self, num_classes=100, drop_path_rate=0.05):
        super().__init__()
        self.model = timm.create_model(
            'deit3_small_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)

    def get_patch_tokens(self, x):
        """Extract patch tokens (excluding CLS) from the final layer."""
        features = self.model.forward_features(x)
        patch_tokens = features[:, 1:, :]
        return patch_tokens

    def get_attention_maps(self, x):
        """Get CLS-to-patch attention maps for each layer via forward hooks."""
        attention_maps = []
        hooks = []

        for block in self.model.blocks:
            storage = []
            attention_maps.append(storage)
            h = block.attn.register_forward_hook(self._make_attn_hook(storage))
            hooks.append(h)

        with torch.no_grad():
            self.model(x)

        for h in hooks:
            h.remove()

        maps = [s[0] for s in attention_maps if len(s) > 0]
        return maps

    @staticmethod
    def _make_attn_hook(storage):
        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            attn = attn.softmax(dim=-1)
            storage.append(attn.detach().cpu())
        return hook_fn


# =============================================================================
# Factory
# =============================================================================

def get_model(args):
    """Factory function to create models by name."""
    if args.model == 'vanilla':
        return VanillaViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
        )
    elif args.model == 'sdpa':
        return SDPViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
        )
    elif args.model == 'register':
        return RegisterViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            num_register_tokens=getattr(args, 'num_register_tokens', 4),
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Available: vanilla, sdpa, register")
