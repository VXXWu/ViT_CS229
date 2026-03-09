"""Additional explorations: pruning curves, attention maps, effective dim, specialization dynamics.

Usage:
  python analyze_extra.py --results-dir ./results --output-dir ./figures \
      --data-path /path/to/archive
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from src.models import ViT

DISPLAY_NAMES = {
    'softmax': 'Softmax',
    'sigmoid': 'Sigmoid',
    'relu': 'ReLU',
    'gated': 'Gated SDPA',
}

VARIANT_COLORS = {
    'softmax': '#3498db',
    'sigmoid': '#e67e22',
    'relu': '#e74c3c',
    'gated': '#2ecc71',
}

VARIANTS = [
    ('softmax', 'softmax', 'layernorm'),
    ('relu', 'relu', 'layernorm'),
    ('sigmoid', 'sigmoid', 'layernorm'),
    ('gated', 'gated', 'layernorm'),
]


def load_model(results_dir, name, attn_type, norm_type, device, ckpt_name='checkpoint_latest.pth'):
    ckpt_path = os.path.join(results_dir, name, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None
    model = ViT(attn_type=attn_type, norm_type=norm_type,
                model_size='small', num_classes=100)
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except (EOFError, RuntimeError):
        return None
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    return model


def load_val_dataset(data_path, batch_size=128):
    from torchvision import datasets, transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    for split in ['val', 'val.X']:
        split_dir = os.path.join(data_path, split)
        if os.path.isdir(split_dir):
            try:
                dataset = datasets.ImageFolder(split_dir, transform=val_transform)
            except FileNotFoundError:
                continue
            if len(dataset) > 0:
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                return loader, dataset
    return None, None


@torch.no_grad()
def evaluate_accuracy(model, loader, device, max_samples=None):
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
        if max_samples and total >= max_samples:
            break
    return correct / total


# ══════════════════════════════════════════════════════════════════════════
# Analysis 1: Multi-head pruning curves
# ══════════════════════════════════════════════════════════════════════════

def multi_head_pruning_curves(results_dir, device, val_loader, output_dir):
    """Progressively prune heads, plot accuracy vs fraction remaining.

    Uses entropy as proxy for head importance — prune lowest-entropy (most
    focused/specialized) heads first to show that specialist heads are
    critical, vs prune highest-entropy (most diffuse) heads first.
    """
    print("Computing attention entropy for pruning order...")
    # First, get per-head entropy to define pruning order
    from torchvision import transforms
    images_batch, _ = next(iter(val_loader))
    images_batch = images_batch[:64].to(device)

    all_results = {}

    for name, attn_type, norm_type in VARIANTS:
        model = load_model(results_dir, name, attn_type, norm_type, device)
        if model is None:
            print(f"  Skipping {name}")
            continue

        num_heads = model.num_heads
        head_dim = model.embed_dim // num_heads
        n_layers = model.depth

        # Get per-head entropy via hooks
        attn_weights_per_layer = []
        def make_hook(attn_mod, storage):
            def hook(module, input, output):
                x_in = input[0]
                B = x_in.shape[0]
                qkv = output.reshape(B, -1, 3, attn_mod.num_heads,
                                     attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                scale = attn_mod.head_dim ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale
                if attn_type in ('softmax', 'gated'):
                    weights = torch.softmax(scores, dim=-1)
                elif attn_type == 'relu':
                    weights = F.relu(scores) / scores.shape[-1]
                elif attn_type == 'sigmoid':
                    weights = torch.sigmoid(scores + attn_mod.attn_bias)
                else:
                    weights = torch.softmax(scores, dim=-1)
                storage.append(weights.cpu())
            return hook

        hooks = []
        for block in model.blocks:
            h = block.attn.qkv.register_forward_hook(
                make_hook(block.attn, attn_weights_per_layer))
            hooks.append(h)
        with torch.no_grad():
            _ = model(images_batch)
        for h in hooks:
            h.remove()

        # Compute per-head entropy
        head_entropies = []  # list of (layer, head, entropy)
        for layer_idx, weights in enumerate(attn_weights_per_layer):
            for h in range(num_heads):
                w = weights[:, h, :, :]
                w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1).mean().item()
                head_entropies.append((layer_idx, h, ent))

        # Sort by entropy: lowest first = most specialized first
        head_entropies.sort(key=lambda x: x[2])
        prune_order_specialist_first = [(l, h) for l, h, _ in head_entropies]
        prune_order_generalist_first = list(reversed(prune_order_specialist_first))

        # Evaluate baseline
        baseline_acc = evaluate_accuracy(model, val_loader, device, max_samples=1000)
        print(f"  {DISPLAY_NAMES[name]:12s}: baseline (1K subset) = {baseline_acc*100:.1f}%")

        # Progressive pruning — evaluate at every 6th step (1 layer's worth)
        steps = list(range(0, len(prune_order_specialist_first) + 1, 6))
        if steps[-1] != len(prune_order_specialist_first):
            steps.append(len(prune_order_specialist_first))

        spec_first_accs = [baseline_acc]
        gen_first_accs = [baseline_acc]

        for n_pruned in steps[1:]:
            # Specialist-first pruning
            active_hooks = []
            for layer_idx, head_idx in prune_order_specialist_first[:n_pruned]:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                def zero_hook(module, input, _s=start, _e=end):
                    x = input[0].clone()
                    x[:, :, _s:_e] = 0
                    return (x,)
                hk = model.blocks[layer_idx].attn.proj.register_forward_pre_hook(zero_hook)
                active_hooks.append(hk)
            acc = evaluate_accuracy(model, val_loader, device, max_samples=1000)
            spec_first_accs.append(acc)
            for hk in active_hooks:
                hk.remove()

            # Generalist-first pruning
            active_hooks = []
            for layer_idx, head_idx in prune_order_generalist_first[:n_pruned]:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                def zero_hook(module, input, _s=start, _e=end):
                    x = input[0].clone()
                    x[:, :, _s:_e] = 0
                    return (x,)
                hk = model.blocks[layer_idx].attn.proj.register_forward_pre_hook(zero_hook)
                active_hooks.append(hk)
            acc = evaluate_accuracy(model, val_loader, device, max_samples=1000)
            gen_first_accs.append(acc)
            for hk in active_hooks:
                hk.remove()

            frac = n_pruned / len(prune_order_specialist_first)
            print(f"    {n_pruned:2d}/{len(prune_order_specialist_first)} pruned ({frac*100:.0f}%): "
                  f"spec-first={spec_first_accs[-1]*100:.1f}%, "
                  f"gen-first={gen_first_accs[-1]*100:.1f}%")

        fractions = [s / len(prune_order_specialist_first) for s in [0] + steps[1:]]
        all_results[name] = {
            'fractions': fractions,
            'specialist_first': spec_first_accs,
            'generalist_first': gen_first_accs,
            'baseline': baseline_acc,
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.set_title('Prune Specialists First\n(low-entropy heads removed first)', fontsize=12)
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot([f * 100 for f in r['fractions']],
                [a * 100 for a in r['specialist_first']],
                'o-', color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('% Heads Pruned', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Random chance')

    ax = axes[1]
    ax.set_title('Prune Generalists First\n(high-entropy heads removed first)', fontsize=12)
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot([f * 100 for f in r['fractions']],
                [a * 100 for a in r['generalist_first']],
                'o-', color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('% Heads Pruned', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('Progressive Head Pruning Curves', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'pruning_curves.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# Analysis 2: Attention map visualization
# ══════════════════════════════════════════════════════════════════════════

def attention_map_visualization(results_dir, data_path, device, output_dir):
    """Visualize CLS→patch attention maps for key heads across models."""
    from torchvision import transforms

    # Load a few sample images (raw, for display)
    val_dir = None
    for split in ['val', 'val.X']:
        d = os.path.join(data_path, split)
        if os.path.isdir(d):
            val_dir = d
            break
    if val_dir is None:
        print("  No val images found")
        return

    # Find sample images from different classes
    classes = sorted(os.listdir(val_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(val_dir, c))]
    sample_images = []
    sample_raw = []
    n_samples = 4

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    raw_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    for i, cls in enumerate(classes[:n_samples]):
        cls_dir = os.path.join(val_dir, cls)
        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
        if imgs:
            img_path = os.path.join(cls_dir, imgs[0])
            img = Image.open(img_path).convert('RGB')
            sample_images.append(preprocess(img))
            sample_raw.append(raw_transform(img))

    if not sample_images:
        print("  No sample images found")
        return

    batch = torch.stack(sample_images).to(device)

    # For each model, extract attention maps at key layers
    key_layers = [0, 5, 11]  # early, mid, late
    n_cols = len(key_layers)
    n_models = 0
    model_names = []
    all_attn_maps = {}

    for name, attn_type, norm_type in VARIANTS:
        model = load_model(results_dir, name, attn_type, norm_type, device)
        if model is None:
            continue
        n_models += 1
        model_names.append(name)

        attn_per_layer = []
        def make_hook(attn_mod, storage, at=attn_type):
            def hook(module, input, output):
                x_in = input[0]
                B = x_in.shape[0]
                qkv = output.reshape(B, -1, 3, attn_mod.num_heads,
                                     attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                scale = attn_mod.head_dim ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale
                if at in ('softmax', 'gated'):
                    weights = torch.softmax(scores, dim=-1)
                elif at == 'relu':
                    weights = F.relu(scores) / scores.shape[-1]
                elif at == 'sigmoid':
                    weights = torch.sigmoid(scores + attn_mod.attn_bias)
                else:
                    weights = torch.softmax(scores, dim=-1)
                storage.append(weights.cpu())
            return hook

        hooks = []
        for block in model.blocks:
            h = block.attn.qkv.register_forward_hook(
                make_hook(block.attn, attn_per_layer))
            hooks.append(h)
        with torch.no_grad():
            _ = model(batch)
        for h in hooks:
            h.remove()

        all_attn_maps[name] = attn_per_layer
        print(f"  {DISPLAY_NAMES[name]}: attention extracted")

    # Plot: rows = models, cols = layers, show CLS→patch attention (head-averaged)
    # Use first sample image
    img_idx = 0
    fig, axes = plt.subplots(n_models, n_cols + 1, figsize=(3.5 * (n_cols + 1), 3.2 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(model_names):
        # Raw image in first column
        axes[row, 0].imshow(sample_raw[img_idx])
        axes[row, 0].set_ylabel(DISPLAY_NAMES[name], fontsize=12, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title('Input', fontsize=11)

        for col, layer_idx in enumerate(key_layers):
            attn = all_attn_maps[name][layer_idx]  # (B, H, N, N)
            # CLS token is index 0, patches are 1:197
            # Average across heads, take CLS row (query=CLS, keys=patches)
            cls_attn = attn[img_idx, :, 0, 1:].mean(dim=0)  # (196,)
            cls_attn = cls_attn.reshape(14, 14).numpy()
            # Normalize for display
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

            axes[row, col + 1].imshow(sample_raw[img_idx], alpha=0.3)
            axes[row, col + 1].imshow(cls_attn, cmap='hot', alpha=0.7,
                                       extent=[0, 224, 224, 0], interpolation='bilinear')
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])
            if row == 0:
                axes[row, col + 1].set_title(f'Layer {layer_idx}', fontsize=11)

    fig.suptitle('CLS → Patch Attention (head-averaged)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'attention_maps.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    # Second figure: per-head attention for ReLU vs Softmax at layer 0
    # (most interesting comparison — ReLU's critical head 1)
    if 'relu' in all_attn_maps and 'softmax' in all_attn_maps:
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for row, name in enumerate(['softmax', 'relu']):
            attn = all_attn_maps[name][0]  # layer 0
            for h in range(6):
                cls_attn = attn[img_idx, h, 0, 1:].reshape(14, 14).numpy()
                cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                axes[row, h].imshow(sample_raw[img_idx], alpha=0.3)
                axes[row, h].imshow(cls_attn, cmap='hot', alpha=0.7,
                                    extent=[0, 224, 224, 0], interpolation='bilinear')
                axes[row, h].set_xticks([])
                axes[row, h].set_yticks([])
                if row == 0:
                    axes[row, h].set_title(f'Head {h}', fontsize=11)
            axes[row, 0].set_ylabel(DISPLAY_NAMES[name], fontsize=12, fontweight='bold')

        fig.suptitle('Layer 0: Per-Head CLS Attention — Softmax vs ReLU', fontsize=14)
        fig.tight_layout()
        path = os.path.join(output_dir, 'attention_maps_layer0_heads.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Analysis 3: Effective dimensionality (PCA)
# ══════════════════════════════════════════════════════════════════════════

def effective_dimensionality(results_dir, output_dir):
    """Compute effective rank of CLS features at each layer via PCA."""
    from sklearn.decomposition import PCA

    all_results = {}

    for name, attn_type, norm_type in VARIANTS:
        feat_path = os.path.join(results_dir, name, 'features', 'epoch199.pt')
        if not os.path.exists(feat_path):
            feat_path = os.path.join(results_dir, name, 'features', 'epoch99.pt')
        if not os.path.exists(feat_path):
            print(f"  Skipping {name}: no features")
            continue

        data = torch.load(feat_path, map_location='cpu', weights_only=False)
        cls_features = data['cls']  # list of 13 tensors, each (5000, 384)
        n_layers = len(cls_features)

        eff_dims_95 = []
        eff_dims_pr = []  # participation ratio
        for layer_idx in range(n_layers):
            X = cls_features[layer_idx].numpy()
            # Center
            X = X - X.mean(axis=0)

            pca = PCA(n_components=min(384, X.shape[0]))
            pca.fit(X)

            # 95% variance threshold
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n95 = np.searchsorted(cumvar, 0.95) + 1
            eff_dims_95.append(n95)

            # Participation ratio: (sum(lambda))^2 / sum(lambda^2)
            eigenvalues = pca.explained_variance_
            pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
            eff_dims_pr.append(pr)

        all_results[name] = {
            'dims_95': eff_dims_95,
            'participation_ratio': eff_dims_pr,
            'n_layers': n_layers,
        }
        print(f"  {DISPLAY_NAMES[name]:12s}: 95% dim = {eff_dims_95[-1]:3d}, "
              f"PR = {eff_dims_pr[-1]:.1f} (final layer)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot(range(r['n_layers']), r['dims_95'], 'o-',
                color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Components for 95% Variance', fontsize=11)
    ax.set_title('Effective Dimensionality (95% Variance Threshold)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot(range(r['n_layers']), r['participation_ratio'], 'o-',
                color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Participation Ratio', fontsize=11)
    ax.set_title('Participation Ratio (Σλ)² / Σλ²', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Feature Space Geometry: CLS Token Representations', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'effective_dimensionality.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# Analysis 4: Specialization dynamics over training
# ══════════════════════════════════════════════════════════════════════════

def specialization_dynamics(results_dir, device, data_path, output_dir):
    """Track head entropy variance (specialization) over training.

    Uses saved checkpoints at different epochs to compute attention entropy.
    """
    from torchvision import datasets, transforms

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_dir = None
    for split in ['val', 'val.X']:
        d = os.path.join(data_path, split)
        if os.path.isdir(d):
            val_dir = d
            break
    if val_dir is None:
        print("  No val images found")
        return

    dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    # Use a small fixed batch
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    images, _ = next(iter(loader))
    images = images.to(device)

    all_results = {}

    for name, attn_type, norm_type in VARIANTS:
        model_dir = os.path.join(results_dir, name)
        # Find all available checkpoints
        ckpts = []
        for f in sorted(os.listdir(model_dir)):
            if f.startswith('checkpoint_epoch') and f.endswith('.pth'):
                epoch = int(f.replace('checkpoint_epoch', '').replace('.pth', ''))
                ckpts.append((epoch, f))
        # Add latest (epoch 199)
        if os.path.exists(os.path.join(model_dir, 'checkpoint_latest.pth')):
            ckpts.append((199, 'checkpoint_latest.pth'))
        ckpts.sort(key=lambda x: x[0])

        if not ckpts:
            print(f"  Skipping {name}: no checkpoints")
            continue

        epochs = []
        entropy_stds = []  # across-head entropy std (specialization metric)
        mean_entropies = []

        for epoch, ckpt_name in ckpts:
            model = load_model(results_dir, name, attn_type, norm_type,
                               device, ckpt_name=ckpt_name)
            if model is None:
                continue

            # Compute attention entropy
            attn_per_layer = []
            def make_hook(attn_mod, storage, at=attn_type):
                def hook(module, input, output):
                    x_in = input[0]
                    B = x_in.shape[0]
                    qkv = output.reshape(B, -1, 3, attn_mod.num_heads,
                                         attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale = attn_mod.head_dim ** -0.5
                    scores = (q @ k.transpose(-2, -1)) * scale
                    if at in ('softmax', 'gated'):
                        weights = torch.softmax(scores, dim=-1)
                    elif at == 'relu':
                        weights = F.relu(scores) / scores.shape[-1]
                    elif at == 'sigmoid':
                        weights = torch.sigmoid(scores + attn_mod.attn_bias)
                    else:
                        weights = torch.softmax(scores, dim=-1)
                    storage.append(weights.cpu())
                return hook

            hooks = []
            for block in model.blocks:
                h = block.attn.qkv.register_forward_hook(
                    make_hook(block.attn, attn_per_layer))
                hooks.append(h)
            with torch.no_grad():
                _ = model(images)
            for h in hooks:
                h.remove()

            # Compute per-head entropy, then std across all heads
            head_ents = []
            for weights in attn_per_layer:
                for h in range(weights.shape[1]):
                    w = weights[:, h, :, :]
                    w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                    ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1).mean().item()
                    head_ents.append(ent)

            epochs.append(epoch)
            entropy_stds.append(np.std(head_ents))
            mean_entropies.append(np.mean(head_ents))
            print(f"  {DISPLAY_NAMES[name]:12s} epoch {epoch:3d}: "
                  f"entropy std = {entropy_stds[-1]:.3f}, mean = {mean_entropies[-1]:.2f}")

            del model
            torch.mps.empty_cache() if device.type == 'mps' else None

        all_results[name] = {
            'epochs': epochs,
            'entropy_std': entropy_stds,
            'mean_entropy': mean_entropies,
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot(r['epochs'], r['entropy_std'], 'o-',
                color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Entropy Std Across Heads', fontsize=11)
    ax.set_title('Head Specialization Over Training', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_results:
            continue
        r = all_results[name]
        ax.plot(r['epochs'], r['mean_entropy'], 'o-',
                color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name],
                linewidth=2, markersize=7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean Attention Entropy', fontsize=11)
    ax.set_title('Mean Attention Entropy Over Training', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Training Dynamics of Head Specialization', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'specialization_dynamics.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--output-dir', type=str, default='./figures')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Analyses to skip: pruning, attn_maps, pca, dynamics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Analysis 3: Effective dimensionality (fastest — no model loading)
    if 'pca' not in args.skip:
        print("=" * 60)
        print("Analysis 3: Effective Dimensionality (PCA)")
        print("=" * 60)
        effective_dimensionality(args.results_dir, args.output_dir)

    # Analysis 2: Attention map visualization
    if 'attn_maps' not in args.skip:
        print("\n" + "=" * 60)
        print("Analysis 2: Attention Map Visualization")
        print("=" * 60)
        attention_map_visualization(args.results_dir, args.data_path,
                                    device, args.output_dir)

    # Analysis 4: Specialization dynamics
    if 'dynamics' not in args.skip:
        print("\n" + "=" * 60)
        print("Analysis 4: Specialization Dynamics")
        print("=" * 60)
        specialization_dynamics(args.results_dir, device, args.data_path,
                                args.output_dir)

    # Analysis 1: Multi-head pruning curves (slowest)
    if 'pruning' not in args.skip:
        print("\n" + "=" * 60)
        print("Analysis 1: Progressive Head Pruning Curves")
        print("=" * 60)
        val_loader, _ = load_val_dataset(args.data_path)
        if val_loader:
            multi_head_pruning_curves(args.results_dir, device, val_loader,
                                       args.output_dir)
        else:
            print("  ERROR: Could not load val dataset")

    print("\nAll analyses complete.")


if __name__ == '__main__':
    main()
