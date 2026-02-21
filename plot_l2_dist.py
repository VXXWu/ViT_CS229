"""Plot L2 norm distributions for vanilla and SDPA-gated models."""

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMS_DIR = os.path.join(BASE_DIR, 'modal_output', 'l2_norms_1e6')
OUT_DIR = os.path.join(BASE_DIR, 'modal_output', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

van = np.load(os.path.join(NORMS_DIR, 'vanilla_epoch199.npy'))
sdpa = np.load(os.path.join(NORMS_DIR, 'sdpa_gated_epoch199.npy'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, norms, label, color in [
    (axes[0], van, 'Vanilla ViT', '#2196F3'),
    (axes[1], sdpa, 'SDPA-Gated ViT', '#FF5722'),
]:
    flat = norms.flatten()
    ax.hist(flat, bins=80, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(flat.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={flat.mean():.2f}')
    ax.set_xlabel('L2 Norm', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{label} — Epoch 199\nmean={flat.mean():.2f}, std={flat.std():.2f}', fontsize=13)
    ax.legend(fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'l2_dist_1e6_epoch199.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
