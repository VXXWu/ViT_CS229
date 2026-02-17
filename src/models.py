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


def get_model(args):
    """Factory function to create models by name."""
    if args.model == 'vanilla':
        return VanillaViT(
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Available: vanilla")
