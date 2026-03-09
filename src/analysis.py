"""CKA and representation analysis across attention mechanism variants."""

import json
import os

import numpy as np
import torch


def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n, p1) tensor
        Y: (n, p2) tensor

    Returns:
        CKA similarity score in [0, 1]
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()

    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10))


def compute_pairwise_cka(features_dict, layer_idx=-1, feature_type='cls'):
    """Compute pairwise CKA between all model variants at a specific layer.

    Args:
        features_dict: {model_name: {'cls': [tensors], 'patch': [tensors]}}
        layer_idx: which layer to compare (-1 = last / post-norm)
        feature_type: 'cls' or 'patch'

    Returns:
        names: list of model names
        cka_matrix: (N, N) numpy array
    """
    names = sorted(features_dict.keys())
    n = len(names)
    cka_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            X = features_dict[names[i]][feature_type][layer_idx].float()
            Y = features_dict[names[j]][feature_type][layer_idx].float()
            # Use same number of samples
            min_n = min(X.shape[0], Y.shape[0])
            cka = linear_cka(X[:min_n], Y[:min_n])
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka

    return names, cka_matrix


def compute_layerwise_cka(features_a, features_b, feature_type='cls'):
    """Compute CKA between two models at every layer pair.

    Args:
        features_a: {'cls': [tensors], 'patch': [tensors]}
        features_b: {'cls': [tensors], 'patch': [tensors]}
        feature_type: 'cls' or 'patch'

    Returns:
        cka_grid: (L_a, L_b) numpy array where L is num_layers + 1 (includes post-norm)
    """
    feats_a = features_a[feature_type]
    feats_b = features_b[feature_type]
    La, Lb = len(feats_a), len(feats_b)
    cka_grid = np.zeros((La, Lb))

    for i in range(La):
        for j in range(Lb):
            X = feats_a[i].float()
            Y = feats_b[j].float()
            min_n = min(X.shape[0], Y.shape[0])
            cka_grid[i, j] = linear_cka(X[:min_n], Y[:min_n])

    return cka_grid


def load_features(results_dir, model_names, epoch='final'):
    """Load saved features from all model runs.

    Args:
        results_dir: path containing per-model subdirectories
        model_names: list of model run directory names
        epoch: 'final' to use last available, or int for specific epoch

    Returns:
        features_dict: {model_name: {'cls': [tensors], 'patch': [tensors]}}
    """
    features_dict = {}
    for name in model_names:
        feat_dir = os.path.join(results_dir, name, 'features')
        if not os.path.isdir(feat_dir):
            print(f"WARNING: No features dir for {name}")
            continue

        # Find the feature file
        if epoch == 'final':
            files = [f for f in os.listdir(feat_dir) if f.endswith('.pt')]
            if not files:
                print(f"WARNING: No feature files for {name}")
                continue
            # Sort numerically by epoch number
            files.sort(key=lambda f: int(''.join(c for c in f if c.isdigit()) or '0'))
            path = os.path.join(feat_dir, files[-1])
        else:
            path = os.path.join(feat_dir, f'epoch{epoch}.pt')

        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            continue

        data = torch.load(path, map_location='cpu', weights_only=False)
        features_dict[name] = {
            'cls': data['cls'],
            'patch': data['patch'],
        }
        print(f"Loaded {name}: {len(data['cls'])} layers, "
              f"{data['cls'][0].shape[0]} samples")

    return features_dict
