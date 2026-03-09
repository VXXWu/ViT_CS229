"""Configurable ViT with multiple attention mechanisms and normalization types.

Attention types:
  - softmax: standard scaled dot-product attention (F.sdpa)
  - sigmoid: sigmoid attention (Ramapuram et al. ICLR 2025)
  - relu: ReLU attention with 1/N scaling (Wortsman et al. 2023)
  - gated: SDPA + sigmoid gate before W_O (Qiu et al. NeurIPS 2025)
  - linear: linear attention with elu+1 kernel (Katharopoulos et al. 2020)

Normalization types:
  - layernorm: standard nn.LayerNorm
  - dyt: Dynamic Tanh (Zhu et al. CVPR 2025)

Supports ViT-S (384/12/6) and ViT-B (768/12/12).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


MODEL_CONFIGS = {
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
}

ATTN_TYPES = ('softmax', 'sigmoid', 'relu', 'gated', 'linear')
NORM_TYPES = ('layernorm', 'dyt')


# ── Normalization ────────────────────────────────────────────────────────────

class DynamicTanh(nn.Module):
    """DyT: replaces LayerNorm with tanh(alpha * x) * weight + bias.

    Zhu, Chen, He, LeCun (CVPR 2025).
    """

    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight + self.bias


def make_norm(dim, norm_type='layernorm'):
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'dyt':
        return DynamicTanh(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ── Attention variants ───────────────────────────────────────────────────────

class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product attention via F.scaled_dot_product_attention."""

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 seq_len=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SigmoidAttention(nn.Module):
    """Sigmoid attention (Ramapuram et al. ICLR 2025, Apple).

    attn = sigmoid(QK^T / sqrt(d) + b) @ V
    b initialized to -log(seq_len) for stable output scale.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 seq_len=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(seq_len)))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.attn_bias
        attn = torch.sigmoid(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ReLUAttention(nn.Module):
    """ReLU attention with 1/N scaling (Wortsman et al. 2023).

    attn = relu(QK^T / sqrt(d)) / N @ V
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 seq_len=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.relu(attn) / N
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedAttention(nn.Module):
    """SDPA with element-wise sigmoid gate (G1 position, Qiu et al. NeurIPS 2025).

    Y' = softmax(QK^T/sqrt(d)) V  *  sigmoid(X W_theta)
    Gate applied before output projection W_O.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 seq_len=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.,
        )

        gate = torch.sigmoid(self.gate(x))
        gate = gate.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_out = attn_out * gate

        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_gate_scores(self, x):
        """Return sigmoid gate scores (B, N, dim) without running full attention."""
        return torch.sigmoid(self.gate(x))


class LinearAttention(nn.Module):
    """Linear attention with elu+1 kernel (Katharopoulos et al. 2020).

    phi(x) = elu(x) + 1
    attn = phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)
    O(N d^2) instead of O(N^2 d).
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 seq_len=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self._phi(q * self.scale)
        k = self._phi(k)

        # Compute phi(K)^T V: (B, H, D, D) via einsum
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        # Compute phi(Q) (phi(K)^T V): (B, H, N, D)
        num = torch.einsum('bhnd,bhde->bhne', q, kv)
        # Normalizer: phi(Q) phi(K)^T 1 → (B, H, N, 1)
        k_sum = k.sum(dim=2)  # (B, H, D)
        denom = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)
        # Avoid division by zero
        x = num / (denom + 1e-6)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


ATTN_CLASSES = {
    'softmax': SoftmaxAttention,
    'sigmoid': SigmoidAttention,
    'relu': ReLUAttention,
    'gated': GatedAttention,
    'linear': LinearAttention,
}


# ── Building blocks ──────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., layer_scale_init=1e-6,
                 attn_type='softmax', norm_type='layernorm', seq_len=197):
        super().__init__()
        self.norm1 = make_norm(dim, norm_type)
        AttnClass = ATTN_CLASSES[attn_type]
        self.attn = AttnClass(dim, num_heads, qkv_bias, attn_drop, drop,
                              seq_len=seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = make_norm(dim, norm_type)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        self.has_ls = layer_scale_init is not None
        if self.has_ls:
            self.ls1 = LayerScale(dim, layer_scale_init)
            self.ls2 = LayerScale(dim, layer_scale_init)
        self.attn_type = attn_type

    def forward(self, x):
        if self.has_ls:
            x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Patch embedding ──────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


# ── ViT ──────────────────────────────────────────────────────────────────────

class ViT(nn.Module):
    """Configurable ViT supporting multiple attention and normalization types.

    Args:
        model_size: 'small' (384/12/6) or 'base' (768/12/12)
        attn_type: 'softmax', 'sigmoid', 'relu', 'gated', 'linear'
        norm_type: 'layernorm' or 'dyt'
        num_registers: number of register tokens (0 = none)
        layer_scale_init: LayerScale init value (None = no LS)
    """

    def __init__(self, model_size='small', num_classes=100,
                 attn_type='softmax', norm_type='layernorm',
                 num_registers=0, layer_scale_init=1e-4,
                 img_size=224, patch_size=16, drop_path_rate=0.1):
        super().__init__()
        cfg = MODEL_CONFIGS[model_size]
        self.embed_dim = cfg['embed_dim']
        self.depth = cfg['depth']
        self.num_heads = cfg['num_heads']
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.attn_type = attn_type
        self.norm_type = norm_type
        self.model_size = model_size

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, self.embed_dim)
        num_patches = self.patch_embed.num_patches
        seq_len = num_patches + 1 + num_registers  # patches + CLS + registers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, self.embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, drop_path=dpr[i],
                  layer_scale_init=layer_scale_init,
                  attn_type=attn_type, norm_type=norm_type,
                  seq_len=seq_len)
            for i in range(self.depth)
        ])

        self.norm = make_norm(self.embed_dim, norm_type)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.num_registers > 0:
            nn.init.normal_(self.register_tokens, std=1e-6)
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
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.num_registers > 0:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])

    def get_patch_tokens(self, x):
        features = self.forward_features(x)
        offset = 1 + self.num_registers
        return features[:, offset:, :]

    def get_layer_features(self, x):
        """Extract CLS and mean-patch features at every layer (for CKA).

        Returns:
            cls_features: list of (B, D) tensors, one per layer + final norm
            patch_features: list of (B, D) tensors (mean-pooled patches)
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.num_registers > 0:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        offset = 1 + self.num_registers
        cls_features = []
        patch_features = []

        for block in self.blocks:
            x = block(x)
            cls_features.append(x[:, 0].detach())
            patch_features.append(x[:, offset:].mean(dim=1).detach())

        x = self.norm(x)
        cls_features.append(x[:, 0].detach())
        patch_features.append(x[:, offset:].mean(dim=1).detach())

        return cls_features, patch_features

    @torch.no_grad()
    def get_gate_scores(self, x):
        """Extract per-layer gate scores. Returns list of (B, N, dim) tensors."""
        if self.attn_type != 'gated':
            return []
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.num_registers > 0:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        gate_scores = []
        for block in self.blocks:
            normed = block.norm1(x)
            g = block.attn.get_gate_scores(normed)
            gate_scores.append(g.detach().cpu())
            x = block(x)
        return gate_scores

    @torch.no_grad()
    def get_attention_entropy(self, x):
        """Compute per-head attention entropy at each layer.

        Returns list of (num_heads,) tensors with mean entropy per head.
        Only meaningful for softmax/sigmoid/relu (not linear).
        """
        if self.attn_type == 'linear':
            return []

        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.num_registers > 0:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        entropies = []
        for block in self.blocks:
            normed = block.norm1(x)
            # Manually compute attention weights
            B_, N, C = normed.shape
            attn_mod = block.attn
            qkv = attn_mod.qkv(normed).reshape(B_, N, 3, attn_mod.num_heads, attn_mod.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = attn_mod.head_dim ** -0.5
            logits = (q @ k.transpose(-2, -1)) * scale

            if self.attn_type == 'softmax' or self.attn_type == 'gated':
                weights = F.softmax(logits, dim=-1)
            elif self.attn_type == 'sigmoid':
                weights = torch.sigmoid(logits + attn_mod.attn_bias)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            elif self.attn_type == 'relu':
                weights = F.relu(logits)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Entropy: -sum(p * log(p)), averaged over queries and batch
            log_w = torch.log(weights + 1e-10)
            entropy = -(weights * log_w).sum(dim=-1)  # (B, H, N)
            mean_entropy = entropy.mean(dim=(0, 2))  # (H,)
            entropies.append(mean_entropy.cpu())

            x = block(x)

        return entropies

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6
