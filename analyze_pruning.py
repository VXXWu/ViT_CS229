"""Head pruning sensitivity analysis + CS229 importance prediction.

Two analyses:
1. Zero out each attention head one at a time, measure accuracy drop on
   ImageNet-100 val set. Reveals which heads carry unique information.
2. Train sklearn regressors to predict head importance from attention statistics
   (entropy, sparsity, max_attn). Connects mechanistic findings to pruning results.

Usage:
  python analyze_pruning.py --results-dir ./results --output-dir ./figures \
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
    """Load a trained ViT checkpoint."""
    ckpt_path = os.path.join(results_dir, name, 'checkpoint_latest.pth')
    if not os.path.exists(ckpt_path):
        return None
    model = ViT(attn_type=attn_type, norm_type=norm_type,
                model_size='small', num_classes=100)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    return model


def load_val_dataset(data_path, batch_size=128):
    """Load ImageNet-100 val set as a DataLoader."""
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
                print(f"  Loaded {len(dataset)} val images from {split_dir}")
                return loader
    return None


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Evaluate top-1 accuracy on a DataLoader."""
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total


def head_pruning_analysis(results_dir, device, val_loader):
    """Compute per-head importance via accuracy drop when each head is zeroed.

    Returns:
        importance: dict {name: (12, 6) ndarray of accuracy drop in pp}
        baselines: dict {name: float baseline accuracy}
    """
    importance = {}
    baselines = {}

    for name, attn_type, norm_type in VARIANTS:
        model = load_model(results_dir, name, attn_type, norm_type, device)
        if model is None:
            print(f"  Skipping {name}: no checkpoint")
            continue

        num_heads = model.num_heads
        head_dim = model.embed_dim // num_heads
        n_layers = model.depth

        baseline_acc = evaluate_accuracy(model, val_loader, device)
        baselines[name] = baseline_acc
        print(f"  {DISPLAY_NAMES[name]:12s}: baseline acc = {baseline_acc*100:.2f}%")

        imp_matrix = np.zeros((n_layers, num_heads))
        t0 = time.time()

        for layer_idx in range(n_layers):
            block = model.blocks[layer_idx]
            for head_idx in range(num_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                def zero_head_hook(module, input, _start=start, _end=end):
                    x = input[0].clone()
                    x[:, :, _start:_end] = 0
                    return (x,)

                hook = block.attn.proj.register_forward_pre_hook(zero_head_hook)
                pruned_acc = evaluate_accuracy(model, val_loader, device)
                hook.remove()

                drop = baseline_acc - pruned_acc
                imp_matrix[layer_idx, head_idx] = drop

            elapsed = time.time() - t0
            eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
            print(f"    layer {layer_idx:2d}/11 done  "
                  f"(max drop this layer: {imp_matrix[layer_idx].max()*100:.2f}pp, "
                  f"ETA: {eta:.0f}s)")

        importance[name] = imp_matrix
        mean_drop = imp_matrix.mean() * 100
        max_drop = imp_matrix.max() * 100
        print(f"  {DISPLAY_NAMES[name]:12s}: mean drop = {mean_drop:.2f}pp, "
              f"max drop = {max_drop:.2f}pp\n")

    return importance, baselines


def compute_attention_stats(results_dir, device, val_loader):
    """Compute per-head attention statistics (entropy, sparsity, max_attn).

    Uses a single batch of real images for attention pattern extraction.
    """
    images, _ = next(iter(val_loader))
    images = images.to(device)

    all_stats = {}
    for name, attn_type, norm_type in VARIANTS:
        model = load_model(results_dir, name, attn_type, norm_type, device)
        if model is None:
            continue

        num_heads = model.num_heads
        head_dim = model.embed_dim // num_heads
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
            _ = model(images)

        for h in hooks:
            h.remove()

        head_entropies = []
        head_sparsities = []
        head_max_attn = []

        for weights in attn_weights_per_layer:
            n_h = weights.shape[1]
            layer_ent, layer_spar, layer_max = [], [], []
            for h in range(n_h):
                w = weights[:, h, :, :]
                w_norm = w / (w.sum(dim=-1, keepdim=True) + 1e-10)
                ent = -(w_norm * (w_norm + 1e-10).log()).sum(dim=-1)
                layer_ent.append(ent.mean().item())
                layer_spar.append((w.abs() < 1e-3).float().mean().item())
                layer_max.append(w.max(dim=-1).values.mean().item())
            head_entropies.append(layer_ent)
            head_sparsities.append(layer_spar)
            head_max_attn.append(layer_max)

        all_stats[name] = {
            'entropy': np.array(head_entropies),
            'sparsity': np.array(head_sparsities),
            'max_attn': np.array(head_max_attn),
        }
        print(f"  {DISPLAY_NAMES[name]:12s}: attn stats computed "
              f"({len(attn_weights_per_layer)} layers x {num_heads} heads)")

    return all_stats


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_importance_heatmaps(importance, baselines, output_dir):
    """Side-by-side heatmaps of per-head accuracy drop."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in importance]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    # Convert to percentage points for readability
    vmax = max((imp * 100).max() for imp in importance.values())
    for ax, name in zip(axes, names):
        imp_pp = importance[name] * 100
        im = ax.imshow(imp_pp, aspect='auto', cmap='YlOrRd', origin='lower',
                       vmin=0, vmax=vmax)
        ax.set_xlabel('Head', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        base_str = f"{baselines[name]*100:.1f}%"
        ax.set_title(f'{DISPLAY_NAMES[name]} ({base_str})', fontsize=12)
        ax.set_xticks(range(imp_pp.shape[1]))
        ax.set_yticks(range(imp_pp.shape[0]))
        fig.colorbar(im, ax=ax, shrink=0.8, label='pp')

    fig.suptitle('Accuracy Drop per Pruned Head (percentage points)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'head_importance_heatmap.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_importance_distribution(importance, output_dir):
    """Bar chart comparing mean/max accuracy drop and per-layer profile."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in importance]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean and max drop bar chart
    ax = axes[0]
    means = [importance[n].mean() * 100 for n in names]
    maxes = [importance[n].max() * 100 for n in names]
    display = [DISPLAY_NAMES[n] for n in names]
    colors = [VARIANT_COLORS[n] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, means, width, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.5, label='Mean drop')
    ax.bar(x + width/2, maxes, width, color=colors, alpha=0.4,
           edgecolor='black', linewidth=0.5, hatch='//', label='Max drop')
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=11)
    ax.set_ylabel('Accuracy Drop (pp)', fontsize=11)
    ax.set_title('Mean vs Max Single-Head Accuracy Drop', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: per-layer mean accuracy drop
    ax = axes[1]
    for name in names:
        layer_mean = importance[name].mean(axis=1) * 100
        ax.plot(range(len(layer_mean)), layer_mean, 'o-',
                label=DISPLAY_NAMES[name], color=VARIANT_COLORS[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Mean Accuracy Drop (pp)', fontsize=11)
    ax.set_title('Per-Layer Head Importance Profile', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'head_importance_distribution.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_importance_variance(importance, output_dir):
    """Per-layer importance std across heads."""
    names = [n for n in ['softmax', 'relu', 'sigmoid', 'gated'] if n in importance]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name in names:
        layer_std = importance[name].std(axis=1) * 100
        ax.plot(range(len(layer_std)), layer_std, 'o-',
                label=DISPLAY_NAMES[name], color=VARIANT_COLORS[name],
                linewidth=2, markersize=5)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Accuracy Drop Std Across Heads (pp)', fontsize=11)
    ax.set_title('Head Specialization via Pruning (Importance Variance by Layer)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'head_importance_variance.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


# ── CS229 regression ──────────────────────────────────────────────────────

def predict_importance_from_attention(importance, attn_stats, output_dir):
    """Train regressors to predict head accuracy drop from attention statistics.

    Two experiments:
    A) Pooled (all models): features = entropy, sparsity, max_attn, layer_idx
       CV = leave-one-model-out. Tests cross-mechanism transfer.
    B) Within-model: for each model separately, 6-fold CV on heads.
       Tests whether attention stats predict importance within a mechanism.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    common = sorted(set(importance.keys()) & set(attn_stats.keys()))
    if len(common) < 2:
        print("Need at least 2 models with both pruning and attention data.")
        return

    n_layers = importance[common[0]].shape[0]
    n_heads = importance[common[0]].shape[1]

    # Build dataset with structural features
    X_all, y_all, model_labels = [], [], []
    for name in common:
        imp = importance[name]  # (12, 6) — accuracy drop
        ent = attn_stats[name]['entropy']
        spar = attn_stats[name]['sparsity']
        maxa = attn_stats[name]['max_attn']
        for l in range(n_layers):
            for h in range(n_heads):
                X_all.append([ent[l, h], spar[l, h], maxa[l, h],
                              l / (n_layers - 1)])
                y_all.append(imp[l, h] * 100)  # in pp
                model_labels.append(name)

    X = np.array(X_all)
    y = np.array(y_all)
    model_labels = np.array(model_labels)
    feat_names = ['Entropy', 'Sparsity', 'Max Attn', 'Layer (norm)']

    print(f"\n  Regression dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Features: {', '.join(feat_names)}")
    print(f"  Target: accuracy drop (pp)")

    regressors = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42,
                                                max_depth=6),
    }

    # ── Experiment A: Leave-one-model-out ──
    cv_results = {rname: {'r2': [], 'models': []} for rname in regressors}

    for held_out in common:
        train_mask = model_labels != held_out
        test_mask = model_labels == held_out

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        y_train, y_test = y[train_mask], y[test_mask]

        for rname, reg in regressors.items():
            reg_clone = type(reg)(**reg.get_params())
            reg_clone.fit(X_train, y_train)
            r2 = r2_score(y_test, reg_clone.predict(X_test))
            cv_results[rname]['r2'].append(r2)
            cv_results[rname]['models'].append(held_out)

    print(f"\n  Experiment A — Leave-one-model-out CV:")
    print(f"  {'Regressor':20s} | {'Mean R²':>8s} | Per-model R²")
    print(f"  {'-'*65}")
    for rname, res in cv_results.items():
        per = ', '.join(f"{DISPLAY_NAMES[m]}={r:.3f}"
                        for m, r in zip(res['models'], res['r2']))
        print(f"  {rname:20s} | {np.mean(res['r2']):>8.3f} | {per}")

    # ── Experiment B: Within-model 6-fold CV ──
    print(f"\n  Experiment B — Within-model 6-fold CV:")
    print(f"  {'Model':15s} | {'Linear':>8s} | {'Ridge':>8s} | {'RF':>8s}")
    print(f"  {'-'*50}")
    within_r2 = {rname: [] for rname in regressors}

    for name in common:
        mask = model_labels == name
        X_m, y_m = X[mask], y[mask]
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        model_scores = {}
        for rname, reg in regressors.items():
            fold_r2 = []
            for tr_idx, te_idx in kf.split(X_m):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_m[tr_idx])
                X_te = scaler.transform(X_m[te_idx])
                reg_clone = type(reg)(**reg.get_params())
                reg_clone.fit(X_tr, y_m[tr_idx])
                r2 = r2_score(y_m[te_idx], reg_clone.predict(X_te))
                fold_r2.append(r2)
            model_scores[rname] = np.mean(fold_r2)
            within_r2[rname].append(np.mean(fold_r2))
        scores_str = ' | '.join(f"{model_scores[r]:>8.3f}" for r in regressors)
        print(f"  {DISPLAY_NAMES[name]:15s} | {scores_str}")

    print(f"  {'Mean':15s} | " + ' | '.join(
        f"{np.mean(within_r2[r]):>8.3f}" for r in regressors))

    # ── Full-data fit for visualization ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    y_pred_pooled = ridge.predict(X_scaled)
    r2_pooled = r2_score(y, y_pred_pooled)

    within_preds = {}
    for name in common:
        mask = model_labels == name
        X_m, y_m = X[mask], y[mask]
        sc = StandardScaler()
        X_ms = sc.fit_transform(X_m)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        rf.fit(X_ms, y_m)
        within_preds[name] = {
            'actual': y_m,
            'predicted': rf.predict(X_ms),
            'r2': r2_score(y_m, rf.predict(X_ms)),
        }

    lr = LinearRegression()
    lr.fit(X_scaled, y)
    print(f"\n  Linear regression coefficients (standardized, pooled):")
    for fn, coef in zip(feat_names, lr.coef_):
        print(f"    {fn}: {coef:+.4f}")

    rf_all = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    rf_all.fit(X_scaled, y)
    print(f"\n  Random Forest feature importance (pooled):")
    for fn, fi in zip(feat_names, rf_all.feature_importances_):
        print(f"    {fn}: {fi:.3f}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for name in common:
        mask = model_labels == name
        ax.scatter(y[mask], y_pred_pooled[mask], alpha=0.5, s=20,
                   color=VARIANT_COLORS[name], label=DISPLAY_NAMES[name])
    lims = [min(y.min(), y_pred_pooled.min()) - 0.5,
            max(y.max(), y_pred_pooled.max()) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Actual Acc Drop (pp)', fontsize=11)
    ax.set_ylabel('Predicted Acc Drop (pp)', fontsize=11)
    ax.set_title(f'Pooled Ridge (R² = {r2_pooled:.3f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in common:
        wp = within_preds[name]
        ax.scatter(wp['actual'], wp['predicted'], alpha=0.5, s=20,
                   color=VARIANT_COLORS[name],
                   label=f"{DISPLAY_NAMES[name]} (R²={wp['r2']:.2f})")
    all_vals = np.concatenate([wp['actual'] for wp in within_preds.values()])
    lims_w = [all_vals.min() - 0.5, all_vals.max() + 0.5]
    ax.plot(lims_w, lims_w, 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Actual Acc Drop (pp)', fontsize=11)
    ax.set_ylabel('Predicted Acc Drop (pp)', fontsize=11)
    ax.set_title('Within-Model RF (train=test)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    reg_names = list(regressors.keys())
    x_pos = np.arange(len(reg_names))
    width = 0.35
    cross_means = [np.mean(cv_results[r]['r2']) for r in reg_names]
    within_means = [np.mean(within_r2[r]) for r in reg_names]
    ax.bar(x_pos - width / 2, cross_means, width, label='Cross-model',
           color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width / 2, within_means, width, label='Within-model',
           color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(reg_names, fontsize=10)
    ax.set_ylabel('Mean R²', fontsize=11)
    ax.set_title('Cross- vs Within-Model Prediction', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    fig.suptitle('Can Attention Statistics Predict Head Importance?', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'importance_prediction.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\n  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--output-dir', type=str, default='./figures')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageNet-100 (with val/ or val.X/ subdir)')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    val_loader = load_val_dataset(args.data_path, args.batch_size)
    if val_loader is None:
        print("ERROR: Could not find val images. Provide --data-path with val/ or val.X/ subdir.")
        return

    # Analysis 1: Head pruning
    print("\n" + "=" * 60)
    print("Analysis 1: Head Pruning Sensitivity (Accuracy)")
    print("=" * 60)
    importance, baselines = head_pruning_analysis(args.results_dir, device, val_loader)

    if importance:
        plot_importance_heatmaps(importance, baselines, args.output_dir)
        plot_importance_distribution(importance, args.output_dir)
        plot_importance_variance(importance, args.output_dir)

    # Compute attention statistics for CS229 regression
    print("\n" + "=" * 60)
    print("Computing Attention Statistics")
    print("=" * 60)
    attn_stats = compute_attention_stats(args.results_dir, device, val_loader)

    # Analysis 2: CS229 regression
    if importance and attn_stats:
        print("\n" + "=" * 60)
        print("Analysis 2: Predicting Head Importance (CS229)")
        print("=" * 60)
        predict_importance_from_attention(importance, attn_stats, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
