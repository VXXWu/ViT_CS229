# CS229 Project: Eliminating High-Norm Artifacts in Vision Transformers

## Project Goal
Investigate and eliminate high-norm token artifacts that emerge in Vision Transformers during training. The project trains ViT-Small on ImageNet-100 following the DeiT-III protocol, then compares baseline DeiT-III against register and gated variants to study how architectural changes affect artifact formation, training stability, and attention sink behavior.

## Key Papers
- **Darcet et al. (ICLR 2024)** — "Vision Transformers Need Registers": identifies high-norm artifacts in ViTs, proposes register tokens to absorb them
- **Bach et al. (TMLR 2025)** — "Registers in Small Vision Transformers": confirms artifacts appear in DeiT-III Small (not just large models)
- **Qiu et al. (2025)** — "Gated Attention for LLMs": head-specific sigmoid gating after SDPA (G1) or value projection (G2) introduces non-linearity, sparsity, and eliminates attention sinks

## Architecture
All models use **DeiT-III ViT-Small** architecture:
- embed_dim=384, depth=12, num_heads=6, patch_size=16
- **LayerScale** (init=1e-5) on both attention and MLP residuals — required for artifact formation at small scale
- Pre-norm (LayerNorm before attention/MLP)
- SDPA (F.scaled_dot_product_attention) for fused attention kernels

## Models
| Model | Class | Description | Params |
|-------|-------|-------------|--------|
| vanilla | VanillaViT | DeiT-III Small baseline (LayerScale, SDPA) | 21.71M |
| register | RegisterViT | + k=4 learnable register tokens (no pos embed) | 21.71M |
| sdpa_gated | SDPAGatedViT | + G1 sigmoid gate after SDPA output (query-dependent) | 23.48M |
| value_gated | ValueGatedViT | + G2 sigmoid gate after value projection (not query-dependent) | 23.48M |

### Gating Details (Qiu et al. 2025)
- Gate mechanism: Y' = Y . sigma(X W_theta) where X = pre-normalized hidden states
- Head-specific, elementwise, multiplicative, sigmoid activation, no bias on W_theta
- G1 (sdpa_gated): gate applied to SDPA output before W_O — query-dependent, sparser (mean ~0.116)
- G2 (value_gated): gate applied to values before SDPA — value-dependent, less sparse (mean ~0.221)
- G1 outperforms G2 due to query-dependent sparsity filtering irrelevant context per-query
- Added params per model: 384x384x12 = 1.77M (gate projections)

## File Structure
cs229_project/
├── train.py                # Local training entry point (argparse)
├── train_modal.py          # Modal cloud training (--detach for reliability)
├── train_colab.ipynb       # Colab notebook
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py           # VanillaViT, RegisterViT, SDPAGatedViT, ValueGatedViT + get_model()
│   ├── data.py             # ThreeAugment, ImageFolder loaders, timm Mixup
│   ├── trainer.py          # DeiT-III training loop, CSV logging, L2 analysis
│   └── analysis.py         # compute_l2_norms, plot_l2_distribution, visualize_attention_maps
├── output/                 # Local training outputs
├── output_old/             # Old vanilla (timm DeiT-III) training results
├── modal_output/           # Downloaded results from Modal
├── archive/                # Raw ImageNet-100 (train.X1-X4, val.X)
└── context.md

## Training Recipe (DeiT-III)
- Optimizer: LAMB (lr configurable, weight_decay=0.05)
- Schedule: cosine with 5 warmup epochs (warmup_lr=1e-6)
- Loss: BCE with label smoothing via timm Mixup
- Augmentation: 3-Augment + ColorJitter(0.3), Mixup=0.8, CutMix=1.0
- AMP (CUDA only), gradient clipping max_norm=1.0
- LayerScale init: 1e-5 (DeiT-III default for ViT-Small)

## Modal Deployment
- GPU: A100-SXM4-40GB, 12 CPUs, 10 DataLoader workers
- Data: imagenet100.tar on volume "imagenet100-data", extracted to local SSD
- Output: volume "vit-checkpoints" (/{model}/ subdirs)
- Use `--detach` for runs that survive local disconnects

## Saved Outputs Per Run
- `metrics.csv`: every epoch (train_loss, val_loss, val_acc1, val_acc5, lr, l2_norm_mean, l2_norm_std, l2_outlier_ratio, epoch_time_s)
- `l2_norms/epoch{N}.npy`: raw L2 norm arrays every epoch
- `checkpoint_epoch{N}.pth`: every 25 epochs + latest
- Plots (l2_dist, attn_maps): every 10 epochs only (can regenerate from checkpoints/norms)

## Key Findings So Far
- Old timm DeiT-III (with LayerScale) showed bimodal L2 norm distribution at epoch 49 — artifacts present
- Custom ViT without LayerScale showed unimodal distribution — no artifacts
- LayerScale is critical for artifact formation at ViT-Small scale (confirmed by Bach et al.)
- All models now use LayerScale to ensure fair comparison and artifact emergence

## Completed Experiments
### SDPA ViT (old, no LayerScale) — 200 epochs, lr=4e-3
- Final: val_acc@1=65.96%, val_acc@5=88.56%
- L2 norm: mean=8.87, std=0.38, outlier_ratio=0.0007
- Results downloaded to modal_output/sdpa/
- Note: this used the old architecture WITHOUT LayerScale

## Useful Commands
```bash
# Run training (detached)
modal run --detach train_modal.py::train --model register --lr 1e-3 --epochs 200

# Check logs
modal app logs <app-id>

# Download results
modal volume get vit-checkpoints /<model>/ ./modal_output/<model>/ --force

# List volume contents
modal volume ls vit-checkpoints /<model>/
```
