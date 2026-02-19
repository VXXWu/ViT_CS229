"""Model definitions for ViT experiments."""

import torch
import torch.nn as nn
import timm


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
        self._attention_maps = []
        self._hooks = []

    def forward(self, x):
        return self.model(x)

    def get_patch_tokens(self, x):
        """Extract patch tokens (excluding CLS) from the final layer."""
        features = self.model.forward_features(x)
        # features shape: (B, N, D) where N = 1 (CLS) + num_patches
        patch_tokens = features[:, 1:, :]  # exclude CLS token
        return patch_tokens

    def _attention_hook(self, module, input, output):
        # timm Attention: output is (B, N, D), but we need the raw attention
        # We hook into the attn_drop layer or compute from qkv
        pass

    def get_attention_maps(self, x):
        """Get CLS-to-patch attention maps for each layer via forward hooks."""
        attention_maps = []

        def make_hook(storage):
            def hook_fn(module, input, output):
                # module is timm's Attention block
                B, N, C = input[0].shape
                qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
                attn = attn.softmax(dim=-1)
                storage.append(attn.detach().cpu())
            return hook_fn

        # Register hooks on each attention block
        hooks = []
        for block in self.model.blocks:
            h = block.attn.register_forward_pre_hook(
                lambda mod, inp, storage=attention_maps: (
                    make_hook(storage)(mod, inp, None) or inp
                )
            )
            hooks.append(h)

        # Actually we need a proper forward hook approach
        # Remove the pre_hooks and use a cleaner method
        for h in hooks:
            h.remove()
        hooks.clear()
        attention_maps.clear()

        for block in self.model.blocks:
            storage = []
            attention_maps.append(storage)
            h = block.attn.register_forward_hook(
                self._make_attn_hook(storage)
            )
            hooks.append(h)

        with torch.no_grad():
            self.model(x)

        for h in hooks:
            h.remove()

        # Each entry in attention_maps is a list with one tensor of shape (B, heads, N, N)
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
    

class RegisterViT(nn.Module):
    """DeiT-III Small with learnable register tokens inserted after CLS."""

    def __init__(self, num_classes=100, drop_path_rate=0.05, num_registers=4):
        super().__init__()
        self.model = timm.create_model(
            'deit3_small_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        self.num_registers = num_registers
        embed_dim = self.model.embed_dim  # 384 for Small
        # Initialise registers with small random values, same scale as patch embed output
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, embed_dim) * 0.02)

    def _forward_features(self, x):
        B = x.shape[0]

        # 1. Patch embedding: (B, 196, 384)
        x = self.model.patch_embed(x)

        # 2. Add pos embed to patches BEFORE prepending CLS
        x = x + self.model.pos_embed

        # 3. Prepend CLS token: (B, 197, 384)
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 4. Inject registers after CLS
        regs = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((x[:, :1], regs, x[:, 1:]), dim=1)

        # 5. Pre-norm, blocks, final norm
        x = self.model.norm_pre(x)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        # Classify from CLS token
        return self.model.head(x[:, 0])

    def get_patch_tokens(self, x):
        """Extract patch tokens only (exclude CLS and registers)."""
        features = self._forward_features(x)
        # Sequence: [CLS, reg_0..reg_{K-1}, patch_0..patch_195]
        patch_tokens = features[:, 1 + self.num_registers:, :]  # (B, 196, 384)
        return patch_tokens

    def get_attention_maps(self, x):
        """Get attention maps per layer, with register rows/cols stripped.

        Returns list of (B, heads, 197, 197) tensors — same shape as VanillaViT
        so analysis.py needs no changes.
        """
        attention_maps = []
        hooks = []

        K = self.num_registers
        # Indices to keep: CLS (0) and all patch cols (1+K .. 196+K)
        keep_idx = [0] + list(range(1 + K, 197 + K))

        for block in self.model.blocks:
            storage = []
            attention_maps.append(storage)
            hooks.append(block.attn.register_forward_hook(
                self._make_attn_hook(storage)
            ))

        with torch.no_grad():
            self._forward_features(x)

        for h in hooks:
            h.remove()

        # Strip register rows and columns before returning
        maps = []
        for storage in attention_maps:
            if len(storage) == 0:
                continue
            attn = storage[0]  # (B, heads, 197+K, 197+K)
            attn = attn[:, :, keep_idx, :][:, :, :, keep_idx]  # (B, heads, 197, 197)
            maps.append(attn)

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

def get_model(args):
    """Factory function to create models by name."""
    if args.model == 'vanilla':
        return VanillaViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
        )
    elif args.model == 'register':
       return RegisterViT(
           num_classes=args.num_classes,
           drop_path_rate=args.drop_path,
           num_registers=args.num_registers,
       )
    else:
        raise ValueError(f"Unknown model: {args.model}. Available: vanilla, register")
