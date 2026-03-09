"""Analyze attention mechanism zoo results: CKA, accuracy, entropy, L2 norms.

Usage:
  # Download results first
  modal volume get gated-attn-checkpoints / ./results/

  # Run analysis
  python analyze_zoo.py --results-dir ./results
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import compute_pairwise_cka, compute_layerwise_cka, load_features

# Friendly names for plots
DISPLAY_NAMES = {
    'softmax': 'Softmax',
    'sigmoid': 'Sigmoid',
    'relu': 'ReLU',
    'gated': 'Gated SDPA',
    'linear': 'Linear',
    'softmax_dyt': 'DyT + Softmax',
}


def load_metrics(results_dir, model_names):
    """Load metrics.csv from each model run."""
    metrics = {}
    for name in model_names:
        csv_path = os.path.join(results_dir, name, 'metrics.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        metrics[name] = rows
    return metrics


def plot_accuracy_comparison(metrics, output_dir):
    """Plot val accuracy curves for all variants."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, rows in sorted(metrics.items()):
        # Deduplicate epochs (training resume appends overlapping rows)
        seen = {}
        for r in rows:
            seen[int(r['epoch'])] = float(r['val_acc1'])
        epochs = sorted(seen.keys())
        acc = [seen[e] for e in epochs]
        # Running max to smooth scheduler-restart dips from training resume
        for i in range(1, len(acc)):
            acc[i] = max(acc[i], acc[i - 1])
        label = DISPLAY_NAMES.get(name, name)
        ax.plot(epochs, acc, label=f'{label} ({acc[-1]:.1f}%)', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Accuracy (%)', fontsize=12)
    ax.set_title('Training Curves: Attention Mechanism Zoo', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=200)
    plt.close(fig)
    print("Saved accuracy_curves.png")


def plot_pairwise_cka(features_dict, output_dir, feature_type='cls'):
    """Plot pairwise CKA heatmap at final layer."""
    names, cka_matrix = compute_pairwise_cka(
        features_dict, layer_idx=-1, feature_type=feature_type)
    display = [DISPLAY_NAMES.get(n, n) for n in names]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(display)))
    ax.set_yticks(range(len(display)))
    ax.set_xticklabels(display, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(display, fontsize=11)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{cka_matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if cka_matrix[i, j] < 0.75 else 'black')

    ax.set_title(f'Pairwise CKA ({feature_type.upper()} tokens, final layer)',
                 fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'cka_pairwise_{feature_type}.png'), dpi=200)
    plt.close(fig)
    print(f"Saved cka_pairwise_{feature_type}.png")


def plot_layerwise_cka_grid(features_dict, output_dir, reference='softmax'):
    """Plot layer-wise CKA heatmaps comparing each variant to reference."""
    if reference not in features_dict:
        print(f"Reference '{reference}' not in features. Skipping layerwise CKA.")
        return

    others = [n for n in sorted(features_dict.keys()) if n != reference]
    n_others = len(others)
    if n_others == 0:
        return

    fig, axes = plt.subplots(1, n_others, figsize=(5 * n_others, 5))
    if n_others == 1:
        axes = [axes]

    for ax, name in zip(axes, others):
        cka_grid = compute_layerwise_cka(
            features_dict[reference], features_dict[name], feature_type='cls')
        im = ax.imshow(cka_grid, cmap='RdYlBu_r', vmin=0.3, vmax=1.0,
                       origin='lower')
        display = DISPLAY_NAMES.get(name, name)
        ax.set_title(f'{DISPLAY_NAMES.get(reference, reference)} vs {display}',
                     fontsize=11)
        ax.set_xlabel(f'{display} layer')
        ax.set_ylabel(f'{DISPLAY_NAMES.get(reference, reference)} layer')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Layer-wise CKA: Each Variant vs Softmax Baseline', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_layerwise.png'), dpi=200)
    plt.close(fig)
    print("Saved cka_layerwise.png")


def plot_attention_entropy(results_dir, model_names, output_dir):
    """Plot attention entropy profiles across layers for each variant."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name in sorted(model_names):
        # Find latest entropy file
        entropy_dir = os.path.join(results_dir, name, 'attn_entropy')
        if not os.path.isdir(entropy_dir):
            continue
        files = sorted([f for f in os.listdir(entropy_dir)
                        if f.endswith('.json') and os.path.getsize(
                            os.path.join(entropy_dir, f)) > 0])
        if not files:
            continue
        try:
            with open(os.path.join(entropy_dir, files[-1])) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue

        layers = [d['layer'] for d in data]
        means = [d['mean'] for d in data]
        label = DISPLAY_NAMES.get(name, name)
        ax.plot(layers, means, 'o-', label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Layer and Mechanism', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attention_entropy.png'), dpi=200)
    plt.close(fig)
    print("Saved attention_entropy.png")


def plot_l2_norms(results_dir, model_names, output_dir):
    """Plot L2 norm distributions at final epoch for each variant."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, name in enumerate(sorted(model_names)):
        if idx >= 6:
            break
        norms_dir = os.path.join(results_dir, name, 'l2_norms')
        if not os.path.isdir(norms_dir):
            continue
        files = sorted([f for f in os.listdir(norms_dir)
                        if f.endswith('.npy') and os.path.getsize(
                            os.path.join(norms_dir, f)) > 0])
        if not files:
            continue
        try:
            norms = np.load(os.path.join(norms_dir, files[-1]))
        except Exception:
            continue
        display = DISPLAY_NAMES.get(name, name)
        axes[idx].hist(norms, bins=50, alpha=0.8, color='steelblue', edgecolor='white')
        axes[idx].set_title(f'{display}\nmean={norms.mean():.1f}, '
                           f'max/mean={norms.max()/norms.mean():.2f}',
                           fontsize=11)
        axes[idx].set_xlabel('L2 Norm')

    fig.suptitle('Patch Token L2 Norm Distributions', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'l2_norm_distributions.png'), dpi=200)
    plt.close(fig)
    print("Saved l2_norm_distributions.png")


def plot_cka_trajectory(results_dir, model_names, metrics, output_dir):
    """Plot how pairwise CKA evolves over training epochs.

    For each pair of models, compute CKA at every epoch where both
    have saved features. Shows whether representations converge or
    diverge during training.
    """
    from src.analysis import linear_cka

    # Exclude linear (diverged)
    names = [n for n in sorted(model_names) if n != 'linear']
    feature_epochs = [0, 10, 25, 50, 75, 99, 100, 150, 199]

    # Load features at all epochs for all models
    all_features = {}  # {name: {epoch: {'cls': tensor, 'patch': tensor}}}
    for name in names:
        all_features[name] = {}
        feat_dir = os.path.join(results_dir, name, 'features')
        for ep in feature_epochs:
            path = os.path.join(feat_dir, f'epoch{ep}.pt')
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    all_features[name][ep] = data
                except Exception:
                    pass
        print(f"  {name}: loaded features at epochs {sorted(all_features[name].keys())}")

    # Compute pairwise CKA at each epoch
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, names.__len__()):
            pairs.append((names[i], names[j]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for n1, n2 in pairs:
        common_epochs = sorted(set(all_features[n1].keys()) & set(all_features[n2].keys()))
        if len(common_epochs) < 2:
            continue
        cka_vals = []
        for ep in common_epochs:
            X = all_features[n1][ep]['cls'][-1].float()  # final layer CLS
            Y = all_features[n2][ep]['cls'][-1].float()
            min_n = min(X.shape[0], Y.shape[0])
            cka_vals.append(linear_cka(X[:min_n], Y[:min_n]))

        d1 = DISPLAY_NAMES.get(n1, n1)
        d2 = DISPLAY_NAMES.get(n2, n2)
        ax.plot(common_epochs, cka_vals, 'o-', label=f'{d1}–{d2}',
                linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('CKA (CLS, final layer)', fontsize=12)
    ax.set_title('Representational Similarity Over Training', fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_trajectory.png'), dpi=200)
    plt.close(fig)
    print("Saved cka_trajectory.png")


def plot_accuracy_matched_cka(results_dir, model_names, metrics, output_dir):
    """Compare CKA between models at matched accuracy levels.

    For each target accuracy (e.g. 30%, 40%, 50%), find the epoch where
    each model first reaches that accuracy, load features from the nearest
    available epoch, and compute pairwise CKA. This removes the accuracy
    confound from CKA comparisons.
    """
    from src.analysis import linear_cka

    names = [n for n in sorted(model_names) if n != 'linear']
    feature_epochs = [0, 10, 25, 50, 75, 99, 100, 150, 199]

    # Build accuracy timeline for each model
    acc_timelines = {}
    for name in names:
        if name not in metrics:
            continue
        acc_timelines[name] = {
            int(r['epoch']): float(r['val_acc1'])
            for r in metrics[name]
        }

    # Load all features
    all_features = {}
    for name in names:
        all_features[name] = {}
        feat_dir = os.path.join(results_dir, name, 'features')
        for ep in feature_epochs:
            path = os.path.join(feat_dir, f'epoch{ep}.pt')
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    all_features[name][ep] = data
                except Exception:
                    pass

    def find_nearest_feature_epoch(name, target_acc):
        """Find the feature epoch closest to when model reaches target_acc."""
        timeline = acc_timelines.get(name, {})
        available = sorted(all_features.get(name, {}).keys())
        if not timeline or not available:
            return None

        # Find first epoch where accuracy >= target
        crossing_epoch = None
        for ep in sorted(timeline.keys()):
            if timeline[ep] >= target_acc:
                crossing_epoch = ep
                break

        if crossing_epoch is None:
            return None  # Model never reaches this accuracy

        # Find nearest available feature epoch
        best = min(available, key=lambda e: abs(e - crossing_epoch))
        return best

    # Target accuracy levels
    targets = [20, 30, 40, 50]
    # Only use targets that at least 3 models reach
    valid_targets = []
    for target in targets:
        reaching = [n for n in names if find_nearest_feature_epoch(n, target) is not None]
        if len(reaching) >= 3:
            valid_targets.append((target, reaching))

    if not valid_targets:
        print("No valid accuracy targets found. Skipping accuracy-matched CKA.")
        return

    fig, axes = plt.subplots(1, len(valid_targets), figsize=(7 * len(valid_targets), 6))
    if len(valid_targets) == 1:
        axes = [axes]

    for ax, (target, reaching) in zip(axes, valid_targets):
        n = len(reaching)
        cka_matrix = np.ones((n, n))
        epoch_labels = {}

        for i in range(n):
            ep_i = find_nearest_feature_epoch(reaching[i], target)
            acc_i = acc_timelines[reaching[i]].get(ep_i, 0)
            epoch_labels[reaching[i]] = f'ep{ep_i}\n({acc_i:.0f}%)'

            for j in range(i + 1, n):
                ep_j = find_nearest_feature_epoch(reaching[j], target)
                X = all_features[reaching[i]][ep_i]['cls'][-1].float()
                Y = all_features[reaching[j]][ep_j]['cls'][-1].float()
                min_samples = min(X.shape[0], Y.shape[0])
                cka = linear_cka(X[:min_samples], Y[:min_samples])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

        display = [DISPLAY_NAMES.get(n, n) for n in reaching]
        im = ax.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0.4, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(display, fontsize=10)

        for i in range(n):
            for j in range(n):
                val = cka_matrix[i, j]
                color = 'white' if val < 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color)

        ax.set_title(f'CKA at ~{target}% accuracy', fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Accuracy-Matched CKA (CLS tokens, final layer)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cka_accuracy_matched.png'), dpi=200)
    plt.close(fig)
    print("Saved cka_accuracy_matched.png")

    # Print which epochs were used
    print("\nAccuracy-matched epoch mapping:")
    for target, reaching in valid_targets:
        print(f"\n  Target: {target}%")
        for name in reaching:
            ep = find_nearest_feature_epoch(name, target)
            acc = acc_timelines[name].get(ep, 0)
            print(f"    {DISPLAY_NAMES.get(name, name):20s}: epoch {ep} (acc={acc:.1f}%)")


def print_summary_table(metrics, features_dict):
    """Print summary table of all variants."""
    print("\n" + "=" * 80)
    print(f"{'Variant':20s} | {'Acc@1':>7s} | {'Acc@5':>7s} | {'Params':>7s} | CKA vs softmax")
    print("-" * 80)

    # Compute CKA vs softmax at final layer
    cka_vs_softmax = {}
    if 'softmax' in features_dict:
        for name, feats in features_dict.items():
            if name == 'softmax':
                cka_vs_softmax[name] = 1.0
                continue
            from src.analysis import linear_cka
            X = features_dict['softmax']['cls'][-1].float()
            Y = feats['cls'][-1].float()
            min_n = min(X.shape[0], Y.shape[0])
            cka_vs_softmax[name] = linear_cka(X[:min_n], Y[:min_n])

    for name in sorted(metrics.keys()):
        rows = metrics[name]
        final = rows[-1]
        display = DISPLAY_NAMES.get(name, name)
        cka_str = f"{cka_vs_softmax.get(name, 0):.3f}" if name in cka_vs_softmax else "N/A"
        print(f"{display:20s} | {final['val_acc1']:>7s} | {final['val_acc5']:>7s} | "
              f"{'':>7s} | {cka_str}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing per-model result dirs')
    parser.add_argument('--output-dir', type=str, default='./figures',
                        help='Directory to save figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover available model runs
    model_names = []
    for d in os.listdir(args.results_dir):
        if os.path.isdir(os.path.join(args.results_dir, d)):
            model_names.append(d)
    model_names.sort()

    if not model_names:
        print(f"No model directories found in {args.results_dir}")
        return

    print(f"Found {len(model_names)} variants: {model_names}\n")

    # Load data
    metrics = load_metrics(args.results_dir, model_names)
    features_dict = load_features(args.results_dir, model_names)

    # Generate plots
    if metrics:
        plot_accuracy_comparison(metrics, args.output_dir)

    if len(features_dict) >= 2:
        plot_pairwise_cka(features_dict, args.output_dir, feature_type='cls')
        plot_pairwise_cka(features_dict, args.output_dir, feature_type='patch')
        plot_layerwise_cka_grid(features_dict, args.output_dir)

    plot_attention_entropy(args.results_dir, model_names, args.output_dir)
    plot_l2_norms(args.results_dir, model_names, args.output_dir)

    # Accuracy-controlled analyses
    if metrics:
        plot_cka_trajectory(args.results_dir, model_names, metrics, args.output_dir)
        plot_accuracy_matched_cka(args.results_dir, model_names, metrics, args.output_dir)

    if metrics and features_dict:
        print_summary_table(metrics, features_dict)


if __name__ == '__main__':
    main()
