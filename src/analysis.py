"""L2 norm tracking and attention map visualization."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def compute_l2_norms(model, images):
    """Compute per-token L2 norms from patch tokens.

    Returns:
        norms: (B*num_patches,) numpy array of L2 norms
        outlier_ratio: fraction of tokens with norm > mean + 3*std
        mean_norm: mean of all token norms
        std_norm: std of all token norms
    """
    patch_tokens = model.get_patch_tokens(images)  # (B, num_patches, D)
    norms = torch.norm(patch_tokens, dim=-1).cpu().numpy().flatten()

    mean_norm = float(norms.mean())
    std_norm = float(norms.std())
    threshold = mean_norm + 3 * std_norm
    outlier_ratio = float((norms > threshold).sum() / len(norms))

    return norms, outlier_ratio, mean_norm, std_norm


def plot_l2_distribution(norms, epoch):
    """Plot histogram of L2 norms."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(norms, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

    mean = norms.mean()
    std = norms.std()
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.2f}')
    ax.axvline(mean + 3 * std, color='orange', linestyle='--', label=f'Mean+3σ={mean + 3*std:.2f}')

    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Count')
    ax.set_title(f'Patch Token L2 Norm Distribution (Epoch {epoch})')
    ax.legend()

    return fig


@torch.no_grad()
def visualize_attention_maps(model, images, epoch, num_images=4):
    """Visualize CLS→patch attention heatmaps per layer.

    Returns a matplotlib figure with a grid: rows=images, cols=layers.
    """
    images = images[:num_images]
    attention_maps = model.get_attention_maps(images)

    if not attention_maps:
        return None

    num_layers = len(attention_maps)
    num_imgs = images.shape[0]

    # Compute number of patches per side
    num_patches = attention_maps[0].shape[-1] - 1  # exclude CLS
    h = w = int(num_patches ** 0.5)

    fig, axes = plt.subplots(num_imgs, num_layers, figsize=(2.5 * num_layers, 2.5 * num_imgs))
    if num_imgs == 1:
        axes = axes[np.newaxis, :]
    if num_layers == 1:
        axes = axes[:, np.newaxis]

    for img_idx in range(num_imgs):
        for layer_idx in range(num_layers):
            attn = attention_maps[layer_idx]  # (B, heads, N, N)
            # Average over heads, take CLS row (row 0), exclude CLS column
            cls_attn = attn[img_idx].mean(dim=0)[0, 1:]  # (num_patches,)
            cls_attn = cls_attn.reshape(h, w).numpy()

            ax = axes[img_idx, layer_idx]
            ax.imshow(cls_attn, cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
            if img_idx == 0:
                ax.set_title(f'Layer {layer_idx}', fontsize=8)

    fig.suptitle(f'CLS→Patch Attention Maps (Epoch {epoch})', fontsize=12)
    fig.tight_layout()
    return fig
