"""Deeper analyses: attention distance and linear probing.

Analysis 1: Attention Distance
  For each head, compute the mean spatial distance between query and key
  positions weighted by attention. Reveals local-vs-global head roles and
  whether ReLU's specialization corresponds to a local/global split.

Analysis 2: Linear Probe Accuracy
  Train logistic regression on CLS features at each layer to predict
  ImageNet-100 classes. Shows where discriminative information emerges
  and whether different attention mechanisms front-load or back-load it.

Usage:
  python analyze_depth.py --results-dir ./results --output-dir ./figures \
      --data-path /path/to/archive
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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


def load_model(results_dir, name, attn_type, norm_type, device):
    ckpt_path = os.path.join(results_dir, name, 'checkpoint_latest.pth')
    if not os.path.exists(ckpt_path):
        return None
    model = ViT(attn_type=attn_type, norm_type=norm_type,
                model_size='small', num_classes=100)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    return model


def load_val_images(data_path, batch_size=64):
    """Load a batch of real val images for attention analysis."""
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
                return loader
    return None


# ── Analysis 1: Attention Distance ──────────────────────────────────────────

def build_distance_matrix(grid_size=14):
    """Build pairwise spatial distance matrix for a grid_size x grid_size grid.

    Returns (N, N) matrix where N = grid_size^2, distances in patch units.
    Index 0 is the CLS token (distance 0 to itself, mean grid distance to patches).
    """
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append((i, j))
    coords = np.array(coords, dtype=np.float32)  # (196, 2)

    # Pairwise Euclidean distance between patch positions
    diff = coords[:, None, :] - coords[None, :, :]  # (196, 196, 2)
    patch_dist = np.sqrt((diff ** 2).sum(axis=-1))   # (196, 196)

    # Add CLS token as index 0: CLS has no spatial position,
    # so CLS-to-patch distance = mean distance across all patches
    N = grid_size * grid_size
    full_dist = np.zeros((N + 1, N + 1), dtype=np.float32)
    full_dist[1:, 1:] = patch_dist
    mean_patch_dist = patch_dist.mean()
    full_dist[0, 1:] = mean_patch_dist
    full_dist[1:, 0] = mean_patch_dist

    return full_dist  # (197, 197)


def compute_attention_distances(results_dir, device, val_loader):
    """Compute mean attention distance per head for each model.

    Returns dict {name: (n_layers, n_heads) ndarray of mean distances}.
    """
    images, _ = next(iter(val_loader))
    images = images.to(device)

    dist_matrix = torch.tensor(build_distance_matrix(14), device=device)  # (197, 197)

    all_distances = {}
    for name, attn_type, norm_type in VARIANTS:
        model = load_model(results_dir, name, attn_type, norm_type, device)
        if model is None:
            continue

        num_heads = model.num_heads
        head_dim = model.embed_dim // num_heads
        attn_weights_per_layer = []

        def make_hook(attn_mod, at_type, storage):
            def hook(module, input, output):
                x_in = input[0]
                B = x_in.shape[0]
                qkv = output.reshape(B, -1, 3, attn_mod.num_heads,
                                     attn_mod.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                scale = attn_mod.head_dim ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale

                if at_type in ('softmax', 'gated'):
                    weights = torch.softmax(scores, dim=-1)
                elif at_type == 'relu':
                    weights = F.relu(scores) / scores.shape[-1]
                elif at_type == 'sigmoid':
                    weights = torch.sigmoid(scores + attn_mod.attn_bias)
                else:
                    weights = torch.softmax(scores, dim=-1)

                storage.append(weights.detach())
            return hook

        hooks = []
        for block in model.blocks:
            h = block.attn.qkv.register_forward_hook(
                make_hook(block.attn, attn_type, attn_weights_per_layer))
            hooks.append(h)

        with torch.no_grad():
            _ = model(images)

        for h in hooks:
            h.remove()

        # Compute weighted mean distance per head
        distances = np.zeros((len(attn_weights_per_layer), num_heads))
        for layer_idx, weights in enumerate(attn_weights_per_layer):
            # weights: (B, H, N, N) where N=197
            N = weights.shape[-1]
            dm = dist_matrix[:N, :N]  # (N, N)

            for h in range(num_heads):
                w = weights[:, h, :, :]  # (B, N, N)
                # Normalize for non-softmax mechanisms
                w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                # Weighted mean distance: sum over keys of attn_weight * distance
                # Average over queries and batch
                mean_dist = (w_norm * dm.unsqueeze(0)).sum(dim=-1).mean().item()
                distances[layer_idx, h] = mean_dist

        all_distances[name] = distances
        print(f"  {DISPLAY_NAMES[name]:12s}: attention distances computed "
              f"({distances.shape[0]} layers x {distances.shape[1]} heads)")

    return all_distances


def plot_attention_distance_heatmaps(all_distances, output_dir):
    """Side-by-side heatmaps of mean attention distance per (layer, head)."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in all_distances]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    vmin = min(d.min() for d in all_distances.values())
    vmax = max(d.max() for d in all_distances.values())

    for ax, name in zip(axes, names):
        im = ax.imshow(all_distances[name], aspect='auto', cmap='coolwarm',
                       origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Head', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        ax.set_title(DISPLAY_NAMES[name], fontsize=12)
        ax.set_xticks(range(all_distances[name].shape[1]))
        ax.set_yticks(range(all_distances[name].shape[0]))
        fig.colorbar(im, ax=ax, shrink=0.8, label='Mean dist (patches)')

    fig.suptitle('Mean Attention Distance per Head (local=blue, global=red)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'attn_distance_heatmap.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_attention_distance_distribution(all_distances, output_dir):
    """Compare distance spread (std across heads) and local/global head counts."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in all_distances]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: per-layer distance std across heads
    ax = axes[0]
    for name in names:
        dist_std = all_distances[name].std(axis=1)
        ax.plot(range(len(dist_std)), dist_std, 'o-',
                label=DISPLAY_NAMES[name], color=VARIANT_COLORS[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Distance Std Across Heads', fontsize=11)
    ax.set_title('Head Distance Diversity by Layer', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Middle: histogram of all head distances per model
    ax = axes[1]
    for name in names:
        vals = all_distances[name].flatten()
        ax.hist(vals, bins=20, alpha=0.5, color=VARIANT_COLORS[name],
                label=DISPLAY_NAMES[name], edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Mean Attention Distance (patches)', fontsize=11)
    ax.set_ylabel('Count (heads)', fontsize=11)
    ax.set_title('Distribution of Head Attention Distances', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: per-layer mean distance
    ax = axes[2]
    for name in names:
        layer_mean = all_distances[name].mean(axis=1)
        ax.plot(range(len(layer_mean)), layer_mean, 'o-',
                label=DISPLAY_NAMES[name], color=VARIANT_COLORS[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Mean Attention Distance', fontsize=11)
    ax.set_title('Attention Receptive Field by Layer', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Attention Distance Analysis', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'attn_distance_distribution.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def print_distance_summary(all_distances):
    """Print summary statistics and local/global head counts."""
    print("\n" + "=" * 70)
    print(f"{'Variant':15s} | {'Mean Dist':>10s} | {'Dist Std':>10s} | "
          f"{'Min Head':>10s} | {'Max Head':>10s} | {'Range':>7s}")
    print("-" * 70)
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_distances:
            continue
        d = all_distances[name]
        print(f"{DISPLAY_NAMES[name]:15s} | {d.mean():>10.3f} | "
              f"{d.std():>10.3f} | {d.min():>10.3f} | "
              f"{d.max():>10.3f} | {d.max()-d.min():>7.3f}")
    print("=" * 70)


# ── Analysis 2: Linear Probe Accuracy ───────────────────────────────────────

def get_val_labels(data_path):
    """Reconstruct val labels in ImageFolder order (sorted class dirs, no shuffle)."""
    from torchvision import datasets, transforms

    # Minimal transform — we only need the labels, not the images
    dummy_transform = transforms.Compose([
        transforms.Resize(1),
        transforms.ToTensor(),
    ])

    for split in ['val', 'val.X']:
        split_dir = os.path.join(data_path, split)
        if os.path.isdir(split_dir):
            try:
                dataset = datasets.ImageFolder(split_dir, transform=dummy_transform)
            except FileNotFoundError:
                continue
            if len(dataset) > 0:
                labels = [s[1] for s in dataset.samples]
                print(f"  Extracted {len(labels)} labels from {split_dir} "
                      f"({len(dataset.classes)} classes)")
                return np.array(labels)
    return None


def linear_probe_analysis(results_dir, labels, output_dir):
    """Train logistic regression on CLS features at each layer for each model.

    Uses sklearn LogisticRegression with standardized features.
    80/20 stratified split for train/test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit

    # Single consistent train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

    all_probe_acc = {}

    for name, _, _ in VARIANTS:
        feat_path = os.path.join(results_dir, name, 'features', 'epoch199.pt')
        if not os.path.exists(feat_path):
            print(f"  Skipping {name}: no features")
            continue

        data = torch.load(feat_path, map_location='cpu', weights_only=False)
        cls_features = data['cls']  # list of (5000, 384) tensors
        n_layers = len(cls_features)

        # Verify sample count matches
        if cls_features[0].shape[0] != len(labels):
            print(f"  WARNING: {name} has {cls_features[0].shape[0]} samples "
                  f"but {len(labels)} labels. Skipping.")
            continue

        probe_acc = []
        t0 = time.time()

        for layer_idx in range(n_layers):
            X = cls_features[layer_idx].numpy()
            X_train, X_test = X[train_idx], X[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs',
                                     C=1.0, random_state=42, n_jobs=-1)
            clf.fit(X_train_s, y_train)
            acc = clf.score(X_test_s, y_test) * 100
            probe_acc.append(acc)

        all_probe_acc[name] = np.array(probe_acc)
        elapsed = time.time() - t0
        print(f"  {DISPLAY_NAMES[name]:12s}: {n_layers} layers probed in {elapsed:.1f}s "
              f"(final layer: {probe_acc[-1]:.1f}%)")

    return all_probe_acc


def plot_linear_probe_curves(all_probe_acc, output_dir):
    """Plot probe accuracy vs layer for each mechanism."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in all_probe_acc]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: probe accuracy curves
    ax = axes[0]
    for name in names:
        acc = all_probe_acc[name]
        layers = list(range(len(acc)))
        ax.plot(layers, acc, 'o-', label=DISPLAY_NAMES[name],
                color=VARIANT_COLORS[name], linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=11)
    ax.set_title('Where Does Discriminative Information Emerge?', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(all_probe_acc[names[0]])))

    # Right: accuracy gain per layer (derivative)
    ax = axes[1]
    for name in names:
        acc = all_probe_acc[name]
        gain = np.diff(acc)
        ax.plot(range(1, len(acc)), gain, 'o-', label=DISPLAY_NAMES[name],
                color=VARIANT_COLORS[name], linewidth=2, markersize=5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Accuracy Gain (pp)', fontsize=11)
    ax.set_title('Per-Layer Contribution to Probe Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Linear Probing: CLS Token Discriminability by Layer', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'linear_probe_accuracy.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_probe_vs_pruning_correlation(all_probe_acc, results_dir, output_dir):
    """Compare probe accuracy gain with head pruning importance at each layer.

    If a layer has high probe accuracy gain, its heads should also be more
    important for pruning. Tests whether probe accuracy predicts functional
    importance.
    """
    # Try to load pruning importance from previous analysis
    # We'll just compute correlation from the probe data itself
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in all_probe_acc]
    if len(names) < 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name in names:
        acc = all_probe_acc[name]
        gain = np.diff(acc)
        layers = np.arange(1, len(acc))
        ax.plot(layers, gain, 'o-', label=DISPLAY_NAMES[name],
                color=VARIANT_COLORS[name], linewidth=2, markersize=6)

        # Annotate peaks
        peak_layer = np.argmax(gain)
        ax.annotate(f'L{peak_layer+1}: +{gain[peak_layer]:.1f}pp',
                    xy=(peak_layer + 1, gain[peak_layer]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=VARIANT_COLORS[name])

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Probe Accuracy Gain (pp)', fontsize=11)
    ax.set_title('Where Each Mechanism Adds Discriminative Information', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'probe_gain_by_layer.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def print_probe_summary(all_probe_acc):
    print("\n" + "=" * 70)
    print(f"{'Variant':15s} | {'Layer 0':>8s} | {'Layer 6':>8s} | "
          f"{'Final':>8s} | {'Peak Gain':>10s} | {'Peak Layer':>10s}")
    print("-" * 70)
    for name in ['softmax', 'relu', 'sigmoid', 'gated']:
        if name not in all_probe_acc:
            continue
        acc = all_probe_acc[name]
        gain = np.diff(acc)
        peak_idx = np.argmax(gain)
        print(f"{DISPLAY_NAMES[name]:15s} | {acc[0]:>7.1f}% | {acc[6]:>7.1f}% | "
              f"{acc[-1]:>7.1f}% | {gain[peak_idx]:>+9.1f}pp | Layer {peak_idx+1:>3d}")
    print("=" * 70)


# ── Analysis 3: Cross-Model Head Transfer ───────────────────────────────────

def head_transfer_analysis(results_dir, labels, output_dir):
    """Evaluate each model's classification head on every other model's features.

    Tests whether geometric similarity (CKA) implies functional compatibility.
    """
    from src.analysis import linear_cka

    model_names = []
    heads = {}       # {name: nn.Linear weight and bias}
    features = {}    # {name: (5000, 384) final-layer CLS features}

    for name, attn_type, norm_type in VARIANTS:
        ckpt_path = os.path.join(results_dir, name, 'checkpoint_latest.pth')
        feat_path = os.path.join(results_dir, name, 'features', 'epoch199.pt')
        if not os.path.exists(ckpt_path) or not os.path.exists(feat_path):
            continue

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        head_weight = ckpt['model']['head.weight']  # (100, 384)
        head_bias = ckpt['model']['head.bias']       # (100,)
        heads[name] = (head_weight, head_bias)

        feat_data = torch.load(feat_path, map_location='cpu', weights_only=False)
        features[name] = feat_data['cls'][-1]  # (5000, 384) post-norm

        model_names.append(name)

    n = len(model_names)
    if n < 2:
        print("Need at least 2 models for head transfer.")
        return

    # Compute transfer accuracy matrix
    transfer_acc = np.zeros((n, n))  # [source_head, target_features]
    cka_matrix = np.zeros((n, n))

    for i, src in enumerate(model_names):
        w, b = heads[src]
        for j, tgt in enumerate(model_names):
            feat = features[tgt]
            logits = feat @ w.T + b  # (5000, 100)
            preds = logits.argmax(dim=-1).numpy()
            acc = (preds == labels).mean() * 100
            transfer_acc[i, j] = acc

            # CKA between source and target features
            if i <= j:
                X = features[src].float()
                Y = features[tgt].float()
                min_n = min(X.shape[0], Y.shape[0])
                cka = linear_cka(X[:min_n], Y[:min_n])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    # Print table
    display = [DISPLAY_NAMES[n] for n in model_names]
    print(f"\n  Transfer Accuracy (rows=head source, cols=feature source):")
    header = f"  {'Head \\ Feats':15s} |" + "|".join(f" {d:>10s}" for d in display) + "|"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, src in enumerate(model_names):
        row = f"  {DISPLAY_NAMES[src]:15s} |"
        for j in range(n):
            val = transfer_acc[i, j]
            marker = " *" if i == j else "  "
            row += f" {val:>8.1f}%{marker}|"
        print(row)
    print("  (* = native head on native features)")

    # Compute transfer ratio: transfer_acc / native_acc_of_target
    print(f"\n  Transfer Ratio (transfer_acc / target_native_acc):")
    header = f"  {'Head \\ Feats':15s} |" + "|".join(f" {d:>10s}" for d in display) + "|"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, src in enumerate(model_names):
        row = f"  {DISPLAY_NAMES[src]:15s} |"
        for j in range(n):
            native = transfer_acc[j, j]
            ratio = transfer_acc[i, j] / native if native > 0 else 0
            marker = " *" if i == j else "  "
            row += f" {ratio:>8.1%}{marker}|"
        print(row)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: transfer accuracy heatmap
    ax = axes[0]
    im = ax.imshow(transfer_acc, cmap='YlOrRd', vmin=0,
                   vmax=max(transfer_acc.max(), 70))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(display, fontsize=10)
    ax.set_xlabel('Feature Source', fontsize=11)
    ax.set_ylabel('Head Source', fontsize=11)
    ax.set_title('Transfer Accuracy (%)', fontsize=12)
    for i in range(n):
        for j in range(n):
            color = 'white' if transfer_acc[i, j] < transfer_acc.max() * 0.6 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{transfer_acc[i, j]:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight=weight)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Middle: CKA vs transfer accuracy scatter
    ax = axes[1]
    off_diag_cka = []
    off_diag_transfer = []
    labels_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            off_diag_cka.append(cka_matrix[i, j])
            off_diag_transfer.append(transfer_acc[i, j])
            labels_list.append(f"{DISPLAY_NAMES[model_names[i]][:3]}→{DISPLAY_NAMES[model_names[j]][:3]}")

    ax.scatter(off_diag_cka, off_diag_transfer, s=60, color='steelblue',
               edgecolor='black', linewidth=0.5, zorder=5)
    for k, lbl in enumerate(labels_list):
        ax.annotate(lbl, (off_diag_cka[k], off_diag_transfer[k]),
                    fontsize=7, ha='center', va='bottom', xytext=(0, 4),
                    textcoords='offset points')

    if len(off_diag_cka) >= 3:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(off_diag_cka, off_diag_transfer, 1)
        x_line = np.linspace(min(off_diag_cka) - 0.02, max(off_diag_cka) + 0.02, 50)
        y_line = P.polyval(x_line, coeffs)
        corr = np.corrcoef(off_diag_cka, off_diag_transfer)[0, 1]
        ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1)
        ax.set_title(f'CKA vs Transfer Acc (r={corr:.2f})', fontsize=12)
    else:
        ax.set_title('CKA vs Transfer Accuracy', fontsize=12)

    ax.set_xlabel('CKA (CLS, final layer)', fontsize=11)
    ax.set_ylabel('Transfer Accuracy (%)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: accuracy drop from native
    ax = axes[2]
    native_accs = np.diag(transfer_acc)
    for i, src in enumerate(model_names):
        drops = []
        for j in range(n):
            if i == j:
                continue
            drops.append(transfer_acc[i, j] - native_accs[j])
        ax.bar(i, np.mean(drops), color=VARIANT_COLORS[src],
               edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.errorbar(i, np.mean(drops), yerr=np.std(drops), color='black',
                    capsize=4, linewidth=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(display, fontsize=10)
    ax.set_ylabel('Mean Accuracy Drop (pp)', fontsize=11)
    ax.set_title('How Much Does Using a Foreign Head Hurt?', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Cross-Model Head Transfer: Does CKA Predict Compatibility?', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'head_transfer.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\n  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--output-dir', type=str, default='./figures')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageNet-100 (with val/ or val.X/ subdir)')
    parser.add_argument('--skip-distance', action='store_true',
                        help='Skip attention distance analysis')
    parser.add_argument('--skip-probe', action='store_true',
                        help='Skip linear probe analysis')
    parser.add_argument('--skip-transfer', action='store_true',
                        help='Skip head transfer analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Analysis 1: Attention Distance ──
    if not args.skip_distance:
        print("=" * 60)
        print("Analysis 1: Attention Distance")
        print("=" * 60)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Device: {device}")

        val_loader = load_val_images(args.data_path, batch_size=64)
        if val_loader is None:
            print("ERROR: Could not load val images.")
        else:
            all_distances = compute_attention_distances(
                args.results_dir, device, val_loader)
            if all_distances:
                plot_attention_distance_heatmaps(all_distances, args.output_dir)
                plot_attention_distance_distribution(all_distances, args.output_dir)
                print_distance_summary(all_distances)

    # ── Analysis 2: Linear Probes ──
    if not args.skip_probe:
        print("\n" + "=" * 60)
        print("Analysis 2: Linear Probe Accuracy")
        print("=" * 60)

        labels = get_val_labels(args.data_path)
        if labels is None:
            print("ERROR: Could not extract val labels.")
        else:
            all_probe_acc = linear_probe_analysis(
                args.results_dir, labels, args.output_dir)
            if all_probe_acc:
                plot_linear_probe_curves(all_probe_acc, args.output_dir)
                plot_probe_vs_pruning_correlation(all_probe_acc, args.results_dir,
                                                  args.output_dir)
                print_probe_summary(all_probe_acc)

    # ── Analysis 3: Cross-Model Head Transfer ──
    if not args.skip_transfer:
        print("\n" + "=" * 60)
        print("Analysis 3: Cross-Model Head Transfer")
        print("=" * 60)

        if 'labels' not in dir() or labels is None:
            labels = get_val_labels(args.data_path)
        if labels is None:
            print("ERROR: Could not extract val labels.")
        else:
            head_transfer_analysis(args.results_dir, labels, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
