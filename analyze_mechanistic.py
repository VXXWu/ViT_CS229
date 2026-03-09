"""Mechanistic analysis of representational divergence between attention variants.

Three analyses:
1. Layer-resolved CLS vs Patch CKA gap (softmax vs ReLU)
2. Layer × Epoch CKA trajectory heatmap
3. Per-head attention sparsity and specialization (requires forward pass)

Usage:
  python analyze_mechanistic.py --results-dir ./results --output-dir ./figures
  python analyze_mechanistic.py --results-dir ./results --output-dir ./figures --run-attention
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import linear_cka

DISPLAY_NAMES = {
    'softmax': 'Softmax',
    'sigmoid': 'Sigmoid',
    'relu': 'ReLU',
    'gated': 'Gated SDPA',
    'linear': 'Linear',
    'softmax_dyt': 'DyT + Softmax',
}

FEATURE_EPOCHS = [0, 10, 25, 50, 75, 99, 100, 150, 199]


def load_all_features(results_dir, names):
    """Load features at all available epochs for given models."""
    all_features = {}
    for name in names:
        all_features[name] = {}
        feat_dir = os.path.join(results_dir, name, 'features')
        for ep in FEATURE_EPOCHS:
            path = os.path.join(feat_dir, f'epoch{ep}.pt')
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    all_features[name][ep] = data
                except Exception:
                    pass
    return all_features


def plot_layer_cka_gap(all_features, output_dir):
    """Analysis 1: Layer-resolved CLS vs Patch CKA gap.

    For softmax-ReLU (accuracy-matched pair), compute CKA at each layer
    for both CLS and patch tokens. The gap reveals where the attention
    mechanism shapes aggregation vs feature extraction.
    """
    pairs = [
        ('softmax', 'relu', 'Softmax vs ReLU (accuracy-matched)'),
        ('softmax', 'sigmoid', 'Softmax vs Sigmoid'),
        ('softmax', 'softmax_dyt', 'Softmax vs DyT (same attention)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (n1, n2, title) in zip(axes, pairs):
        if n1 not in all_features or n2 not in all_features:
            continue
        if 199 not in all_features[n1] or 199 not in all_features[n2]:
            continue

        feats1 = all_features[n1][199]
        feats2 = all_features[n2][199]
        n_layers = len(feats1['cls'])

        cls_cka = []
        patch_cka = []
        for layer in range(n_layers):
            X_cls = feats1['cls'][layer].float()
            Y_cls = feats2['cls'][layer].float()
            min_n = min(X_cls.shape[0], Y_cls.shape[0])
            cls_cka.append(linear_cka(X_cls[:min_n], Y_cls[:min_n]))

            X_patch = feats1['patch'][layer].float()
            Y_patch = feats2['patch'][layer].float()
            min_n = min(X_patch.shape[0], Y_patch.shape[0])
            patch_cka.append(linear_cka(X_patch[:min_n], Y_patch[:min_n]))

        layers = list(range(n_layers))
        gap = [p - c for p, c in zip(patch_cka, cls_cka)]

        ax.plot(layers, cls_cka, 'o-', label='CLS token', linewidth=2,
                markersize=5, color='#e74c3c')
        ax.plot(layers, patch_cka, 's-', label='Patch tokens (mean)',
                linewidth=2, markersize=5, color='#3498db')
        ax.fill_between(layers, cls_cka, patch_cka, alpha=0.15, color='gray')
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('CKA', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.05)
        ax.set_xticks(layers)

        # Annotate max gap
        max_gap_idx = np.argmax(gap)
        ax.annotate(f'gap={gap[max_gap_idx]:.3f}',
                    xy=(max_gap_idx, (cls_cka[max_gap_idx] + patch_cka[max_gap_idx]) / 2),
                    fontsize=8, ha='center', color='gray')

    fig.suptitle('Where Do Representations Diverge? CLS vs Patch CKA by Layer',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_cls_vs_patch_by_layer.png'), dpi=200)
    plt.close(fig)
    print("Saved cka_cls_vs_patch_by_layer.png")


def plot_layer_epoch_heatmap(all_features, output_dir):
    """Analysis 2: Layer × Epoch CKA heatmap.

    For softmax-ReLU, compute CKA at each (layer, epoch) combination.
    Shows exactly where and when representations diverge.
    """
    pairs = [
        ('softmax', 'relu'),
        ('softmax', 'gated'),
        ('softmax', 'softmax_dyt'),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (n1, n2) in zip(axes, pairs):
        common_epochs = sorted(
            set(all_features.get(n1, {}).keys()) &
            set(all_features.get(n2, {}).keys())
        )
        if len(common_epochs) < 2:
            continue

        n_layers = len(all_features[n1][common_epochs[0]]['cls'])
        heatmap = np.zeros((n_layers, len(common_epochs)))

        for j, ep in enumerate(common_epochs):
            for i in range(n_layers):
                X = all_features[n1][ep]['cls'][i].float()
                Y = all_features[n2][ep]['cls'][i].float()
                min_n = min(X.shape[0], Y.shape[0])
                heatmap[i, j] = linear_cka(X[:min_n], Y[:min_n])

        im = ax.imshow(heatmap, aspect='auto', cmap='RdYlBu_r',
                       vmin=0.3, vmax=1.0, origin='lower')
        ax.set_xticks(range(len(common_epochs)))
        ax.set_xticklabels(common_epochs, fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        d1 = DISPLAY_NAMES.get(n1, n1)
        d2 = DISPLAY_NAMES.get(n2, n2)
        ax.set_title(f'{d1} vs {d2}', fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('CKA by Layer and Training Epoch (CLS tokens)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_layer_epoch_heatmap.png'), dpi=200)
    plt.close(fig)
    print("Saved cka_layer_epoch_heatmap.png")


def analyze_attention_patterns(results_dir, output_dir, data_path=None):
    """Analysis 3: Per-head attention sparsity and specialization.

    Loads softmax and ReLU checkpoints, runs a forward pass on val data
    to extract attention weights, then computes per-head statistics.
    """
    from src.models import ViT
    from torchvision import datasets, transforms

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load checkpoints
    variants = {}
    for name, attn_type in [('softmax', 'softmax'), ('relu', 'relu'),
                             ('sigmoid', 'sigmoid'), ('gated', 'gated')]:
        ckpt_path = os.path.join(results_dir, name, 'checkpoint_latest.pth')
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {name}: no checkpoint")
            continue
        model = ViT(attn_type=attn_type, model_size='small', num_classes=100)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval().to(device)
        variants[name] = model
        print(f"  Loaded {name} checkpoint")

    if len(variants) < 2:
        print("Need at least 2 model checkpoints. Skipping attention analysis.")
        return

    # Create a simple val dataset (ImageNet-100 val)
    if data_path and os.path.isdir(os.path.join(data_path, 'val')):
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        val_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'val'), transform=val_transform)
    else:
        print("  No data_path provided. Using random input for attention analysis.")
        val_dataset = None

    # Generate input batch
    if val_dataset:
        loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        images, _ = next(iter(loader))
    else:
        images = torch.randn(64, 3, 224, 224)
    images = images.to(device)

    # Extract attention weights using hooks on QKV computation
    all_attn_stats = {}
    for name, model in variants.items():
        attn_weights_per_layer = []
        head_entropies = []
        head_sparsities = []
        head_max_attn = []

        def make_hook(attn_type, attn_module, storage):
            def hook(module, input, output):
                x_in = input[0]
                B = x_in.shape[0]
                head_dim = attn_module.head_dim
                scale = head_dim ** -0.5
                qkv = output.reshape(B, -1, 3, attn_module.num_heads,
                                     head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                scores = (q @ k.transpose(-2, -1)) * scale

                if attn_type in ('softmax', 'gated'):
                    weights = torch.softmax(scores, dim=-1)
                elif attn_type == 'relu':
                    weights = torch.relu(scores) / scores.shape[-1]
                elif attn_type == 'sigmoid':
                    weights = torch.sigmoid(scores + attn_module.attn_bias)
                else:
                    weights = torch.softmax(scores, dim=-1)

                storage.append(weights.cpu())
            return hook

        hooks = []
        for block in model.blocks:
            attn = block.attn
            attn_type = name
            h = attn.qkv.register_forward_hook(
                make_hook(attn_type, attn, attn_weights_per_layer))
            hooks.append(h)

        with torch.no_grad():
            _ = model(images)

        for h in hooks:
            h.remove()

        # Compute per-head statistics
        for layer_idx, weights in enumerate(attn_weights_per_layer):
            # weights: (B, num_heads, N, N)
            n_heads = weights.shape[1]
            layer_entropy = []
            layer_sparsity = []
            layer_max = []

            for h in range(n_heads):
                w = weights[:, h, :, :]  # (B, N, N)

                # Entropy per query token, averaged
                # For softmax: weights sum to 1, so entropy is well-defined
                # For relu/sigmoid: normalize per-row first for entropy calc
                w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1)  # (B, N)
                layer_entropy.append(ent.mean().item())

                # Sparsity: fraction of attention weights < 1e-3
                layer_sparsity.append((w.abs() < 1e-3).float().mean().item())

                # Max attention weight (averaged across batch and queries)
                layer_max.append(w.max(dim=-1).values.mean().item())

            head_entropies.append(layer_entropy)
            head_sparsities.append(layer_sparsity)
            head_max_attn.append(layer_max)

        all_attn_stats[name] = {
            'entropy': np.array(head_entropies),      # (n_layers, n_heads)
            'sparsity': np.array(head_sparsities),     # (n_layers, n_heads)
            'max_attn': np.array(head_max_attn),       # (n_layers, n_heads)
        }
        print(f"  {name}: {len(attn_weights_per_layer)} layers, "
              f"{attn_weights_per_layer[0].shape[1]} heads")

    # Plot 1: Per-head entropy heatmaps
    n_variants = len(all_attn_stats)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]

    for ax, (name, stats) in zip(axes, sorted(all_attn_stats.items())):
        im = ax.imshow(stats['entropy'], aspect='auto', cmap='viridis',
                       origin='lower')
        ax.set_xlabel('Head', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        ax.set_title(f'{DISPLAY_NAMES.get(name, name)}\nPer-Head Entropy',
                     fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Attention Entropy by Layer and Head', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attn_per_head_entropy.png'), dpi=200)
    plt.close(fig)
    print("Saved attn_per_head_entropy.png")

    # Plot 2: Per-head sparsity heatmaps
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5))
    if n_variants == 1:
        axes = [axes]

    for ax, (name, stats) in zip(axes, sorted(all_attn_stats.items())):
        im = ax.imshow(stats['sparsity'], aspect='auto', cmap='YlOrRd',
                       origin='lower', vmin=0, vmax=1)
        ax.set_xlabel('Head', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        ax.set_title(f'{DISPLAY_NAMES.get(name, name)}\nSparsity (frac < 1e-3)',
                     fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Attention Sparsity by Layer and Head', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attn_per_head_sparsity.png'), dpi=200)
    plt.close(fig)
    print("Saved attn_per_head_sparsity.png")

    # Plot 3: Head specialization — entropy variance across heads per layer
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, stats in sorted(all_attn_stats.items()):
        entropy_std = stats['entropy'].std(axis=1)  # std across heads
        ax.plot(range(len(entropy_std)), entropy_std, 'o-',
                label=DISPLAY_NAMES.get(name, name), linewidth=2, markersize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Entropy Std Across Heads', fontsize=12)
    ax.set_title('Head Specialization: Entropy Variance by Layer', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attn_head_specialization.png'), dpi=200)
    plt.close(fig)
    print("Saved attn_head_specialization.png")

    # Print summary stats
    print("\n" + "=" * 70)
    print(f"{'Variant':15s} | {'Mean Entropy':>13s} | {'Mean Sparsity':>14s} | "
          f"{'Entropy Std':>12s} | {'Max Attn':>9s}")
    print("-" * 70)
    for name, stats in sorted(all_attn_stats.items()):
        print(f"{DISPLAY_NAMES.get(name, name):15s} | "
              f"{stats['entropy'].mean():>13.3f} | "
              f"{stats['sparsity'].mean():>14.4f} | "
              f"{stats['entropy'].std(axis=1).mean():>12.3f} | "
              f"{stats['max_attn'].mean():>9.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--output-dir', type=str, default='./figures')
    parser.add_argument('--run-attention', action='store_true',
                        help='Run attention pattern analysis (needs checkpoints)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to ImageNet-100 data for attention analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = [n for n in ['softmax', 'sigmoid', 'relu', 'gated', 'softmax_dyt']
             if os.path.isdir(os.path.join(args.results_dir, n, 'features'))]
    print(f"Found variants: {names}")

    # Load all features
    print("\nLoading features at all epochs...")
    all_features = load_all_features(args.results_dir, names)
    for name in names:
        epochs = sorted(all_features.get(name, {}).keys())
        print(f"  {name}: {len(epochs)} epochs")

    # Analysis 1: Layer-resolved CLS vs Patch gap
    print("\n--- Analysis 1: CLS vs Patch CKA by Layer ---")
    plot_layer_cka_gap(all_features, args.output_dir)

    # Analysis 2: Layer × Epoch heatmap
    print("\n--- Analysis 2: Layer × Epoch CKA Heatmap ---")
    plot_layer_epoch_heatmap(all_features, args.output_dir)

    # Analysis 3: Attention patterns (optional, needs checkpoints)
    if args.run_attention:
        print("\n--- Analysis 3: Per-Head Attention Patterns ---")
        analyze_attention_patterns(args.results_dir, args.output_dir,
                                   data_path=args.data_path)


if __name__ == '__main__':
    main()
