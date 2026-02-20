"""Model definitions for ViT experiments (DeiT-III architecture)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


# =============================================================================
# Shared components
# =============================================================================

class LayerScale(nn.Module):
    """Per-channel learnable scaling on residual branches (Touvron et al., 2021).

    DeiT-III uses init_values=1e-4 for ViT-Small.
    """

    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


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


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Blocks (DeiT-III: pre-norm + LayerScale)
# =============================================================================

class Block(nn.Module):
    """Standard DeiT-III transformer block with LayerScale."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., layer_scale_init=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.ls2 = LayerScale(dim, layer_scale_init)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class SDPAGatedBlock(nn.Module):
    """DeiT-III block with SDPA output gating (G1) and LayerScale."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., layer_scale_init=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPAGatedAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.ls2 = LayerScale(dim, layer_scale_init)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ValueGatedBlock(nn.Module):
    """DeiT-III block with value gating (G2) and LayerScale."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., layer_scale_init=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ValueGatedAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.ls2 = LayerScale(dim, layer_scale_init)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


# =============================================================================
# Gated attention modules (Qiu et al. 2025)
# =============================================================================

class SDPAGatedAttention(nn.Module):
    """Multi-head self-attention with sigmoid gate after SDPA (G1 position).

    Implements Eq. 5 from Qiu et al. 2025: Y' = Y . sigma(X W_theta)
    where Y = SDPA output and X = pre-normalized hidden states.
    Gate is query-dependent: each token's gate score depends on its own
    hidden state X_i, introducing non-linearity between W_V and W_O
    (Eq. 8) and input-dependent sparsity.
    """

    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.,
        )

        gate = torch.sigmoid(self.gate(x))
        gate = gate.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_out = attn_out * gate

        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ValueGatedAttention(nn.Module):
    """Multi-head self-attention with sigmoid gate after value projection (G2).

    Implements Eq. 5 at G2 position: V' = V . sigma(X W_theta)
    where V = X W_V and X = pre-normalized hidden states.
    Gate is NOT query-dependent: each token j's gate depends on X_j (Eq. 7).
    """

    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        gate = torch.sigmoid(self.gate(x))
        gate = gate.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v * gate

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Baseline DeiT-III ViT-Small
# =============================================================================

class VanillaViT(nn.Module):
    """DeiT-III ViT-Small baseline.

    ViT-Small config: embed_dim=384, depth=12, num_heads=6, patch_size=16.
    Includes LayerScale (init=1e-4) on both attention and MLP residuals.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05, layer_scale_init=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i], layer_scale_init)
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
        return self.head(x[:, 0])

    def get_patch_tokens(self, x):
        features = self.forward_features(x)
        return features[:, 1:, :]

    def get_attention_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

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
# Register ViT (DeiT-III + learnable register tokens)
# =============================================================================

class RegisterViT(nn.Module):
    """DeiT-III ViT-Small with register tokens (Darcet et al., 2024).

    Adds k learnable register tokens that participate in attention alongside
    CLS and patch tokens. Register tokens are discarded before the
    classification head. They absorb high-norm artifacts that would otherwise
    appear in patch tokens.

    Token layout: [CLS, register_1, ..., register_k, patch_1, ..., patch_196]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05, layer_scale_init=1e-4,
                 num_register_tokens=4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i], layer_scale_init)
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
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        reg = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x[:, 0])

    def get_patch_tokens(self, x):
        features = self.forward_features(x)
        return features[:, 1 + self.num_register_tokens:, :]

    def get_register_tokens(self, x):
        features = self.forward_features(x)
        return features[:, 1:1 + self.num_register_tokens, :]

    def get_attention_maps(self, x):
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
# SDPA-Gated ViT (DeiT-III + G1 gate after SDPA output)
# Qiu et al. 2025
# =============================================================================

class SDPAGatedViT(nn.Module):
    """DeiT-III ViT-Small with sigmoid gate after SDPA output (G1).

    Each attention layer applies a head-specific elementwise sigmoid gate
    to the SDPA output before the output projection. The gate score is
    query-dependent: sigma(X_i W_theta) for token i (Eq. 8). This breaks
    the low-rank bottleneck between W_V and W_O, introduces input-dependent
    sparsity, and mitigates attention sinks.

    Added params: dim x dim per layer (no bias) = 384x384x12 ~ 1.77M for ViT-S.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05, layer_scale_init=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            SDPAGatedBlock(embed_dim, num_heads, mlp_ratio, qkv_bias,
                           drop_rate, attn_drop_rate, dpr[i], layer_scale_init)
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
        return self.head(x[:, 0])

    def get_patch_tokens(self, x):
        features = self.forward_features(x)
        return features[:, 1:, :]

    def get_attention_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

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

    def get_gate_scores(self, x):
        """Extract per-layer gate scores sigma(X W_theta) for analysis."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        gate_scores = []
        for block in self.blocks:
            normed = block.norm1(x)
            g = torch.sigmoid(block.attn.gate(normed))
            gate_scores.append(g.detach().cpu())
            x = block(x)

        return gate_scores


# =============================================================================
# Value-Gated ViT (DeiT-III + G2 gate after value projection)
# Qiu et al. 2025
# =============================================================================

class ValueGatedViT(nn.Module):
    """DeiT-III ViT-Small with sigmoid gate after value projection (G2).

    Each attention layer applies a head-specific elementwise sigmoid gate
    to the value vectors before SDPA. The gate score depends on X_j (the
    token being attended to), not the query (Eq. 7). This introduces
    non-linearity but produces higher (less sparse) gate scores than G1.

    Added params: dim x dim per layer (no bias) = 384x384x12 ~ 1.77M for ViT-S.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.05, layer_scale_init=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            ValueGatedBlock(embed_dim, num_heads, mlp_ratio, qkv_bias,
                            drop_rate, attn_drop_rate, dpr[i], layer_scale_init)
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
        return self.head(x[:, 0])

    def get_patch_tokens(self, x):
        features = self.forward_features(x)
        return features[:, 1:, :]

    def get_attention_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

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

    def get_gate_scores(self, x):
        """Extract per-layer gate scores sigma(X W_theta) for analysis."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        gate_scores = []
        for block in self.blocks:
            normed = block.norm1(x)
            g = torch.sigmoid(block.attn.gate(normed))
            gate_scores.append(g.detach().cpu())
            x = block(x)

        return gate_scores


# =============================================================================
# Factory
# =============================================================================

def get_model(args):
    """Factory function to create models by name."""
    ls_init = getattr(args, 'layer_scale_init', 1e-4)
    if args.model == 'vanilla':
        return VanillaViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init=ls_init,
        )
    elif args.model == 'register':
        return RegisterViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            num_register_tokens=getattr(args, 'num_register_tokens', 4),
            layer_scale_init=ls_init,
        )
    elif args.model == 'sdpa_gated':
        return SDPAGatedViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init=ls_init,
        )
    elif args.model == 'value_gated':
        return ValueGatedViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init=ls_init,
        )
    else:
        raise ValueError(
            f"Unknown model: {args.model}. "
            "Available: vanilla, register, sdpa_gated, value_gated"
        )
