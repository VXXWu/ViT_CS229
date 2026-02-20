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
- **LayerScale** (init=1e-4) on both attention and MLP residuals — required for artifact formation at small scale
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
├── run_resilient.sh        # Auto-restart wrapper for local training
├── train_colab.ipynb       # Colab notebook
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py           # VanillaViT, RegisterViT, SDPAGatedViT, ValueGatedViT + get_model()
│   ├── data.py             # ThreeAugment, ImageFolder loaders, timm Mixup
│   ├── trainer.py          # DeiT-III training loop, CSV logging, L2 analysis
│   └── analysis.py         # compute_l2_norms, plot_l2_distribution, visualize_attention_maps
├── output/                 # Local training outputs
├── output_old/             # Old vanilla (timm DeiT-III, broken optimizer) training results
├── modal_output/           # Downloaded results from Modal
├── archive/                # Raw ImageNet-100 (train.X1-X4, val.X)
└── context.md

## Training Recipe (DeiT-III)
- Optimizer: LAMB via `timm.optim.create_optimizer_v2` with proper param group splitting
  - 1D params (LayerScale gamma, LayerNorm, biases) → weight_decay=0, no LAMB trust ratio (Adam-like updates)
  - 2D weight matrices → weight_decay=0.05, full LAMB trust ratio adaptation
- Schedule: cosine with 5 warmup epochs (warmup_lr=1e-6)
- Loss: BCE with label smoothing via timm Mixup
- Augmentation: 3-Augment + ColorJitter(0.3), Mixup=0.8, CutMix=1.0
- AMP (CUDA only), gradient clipping max_norm=1.0
- LayerScale init: 1e-4 (DeiT-III default)
- Target: 400 epochs (official DeiT-III uses 800 on ImageNet-1K)

## Modal Deployment
- Two entry points: `train` (H100, 12 CPUs, 10 workers) and `train_fast` (H100, 20 CPUs, 16 workers)
- Data: imagenet100.tar on volume "imagenet100-data", extracted to local SSD at /tmp/
- Output: volume "vit-checkpoints" (/{run_name}/ subdirs, where run_name = {model}_{suffix} if suffix provided)
- Use `--detach` for runs that survive local disconnects
- `--suffix` flag to customize output subdir (e.g., `--suffix ls1e5` → `/output/vanilla_ls1e5/`)
- `--layer-scale-init` flag to override LayerScale init value (default 1e-4)
- Multiple runs can execute in parallel on separate containers with no GPU contention

## Saved Outputs Per Run
- `metrics.csv`: every epoch (train_loss, val_loss, val_acc1, val_acc5, lr, l2_norm_mean, l2_norm_std, l2_outlier_ratio, epoch_time_s)
- `l2_norms/epoch{N}.npy`: raw L2 norm arrays every epoch
- `checkpoint_epoch{N}.pth`: every 25 epochs + latest
- Plots (l2_dist, attn_maps): can regenerate from checkpoints/norms

## Key Findings So Far
- Old timm DeiT-III (with LayerScale) showed bimodal L2 norm distribution at epoch 49 — artifacts present
- Custom ViT without LayerScale showed unimodal distribution — no artifacts
- LayerScale is critical for artifact formation at ViT-Small scale (confirmed by Bach et al.)
- All models now use LayerScale to ensure fair comparison and artifact emergence

### LAMB + LayerScale Optimizer Bug (fixed)
Previous training used `Lamb(model.parameters(), ...)` which applied weight_decay=0.05 and
LAMB trust ratio to ALL parameters, including LayerScale gamma (init=1e-5 at the time).
LAMB trust ratio = ||param|| / ||grad||; with gamma ~1e-5, ||param|| ≈ 2e-4, so effective lr ≈ 8e-7.
This starved LayerScale — blocks never "opened up", causing ~20% accuracy plateau at epoch 150+.
Fix: use `timm.optim.create_optimizer_v2` which splits 1D params into weight_decay=0 group;
LAMB skips trust ratio for wd=0 params, giving them Adam-like updates. Also changed init from 1e-5 to 1e-4.

## Completed Experiments
### SDPA ViT (old, no LayerScale) — 200 epochs, lr=4e-3
- Final: val_acc@1=65.96%, val_acc@5=88.56%
- L2 norm: mean=8.87, std=0.38, outlier_ratio=0.0007
- Results downloaded to modal_output/sdpa/
- Note: this used the old architecture WITHOUT LayerScale

### Old Vanilla DeiT-III (broken optimizer) — 157 epochs, lr=4e-3
- Final: val_acc@1=~24%, stuck in plateau
- Results in output_old/metrics.csv
- Note: suffered from LAMB + LayerScale optimizer bug described above

## Current Training
- Model: vanilla (DeiT-III Small, LayerScale init=1e-4, fixed optimizer)
- Epochs: 400
- GPU: RTX 4090, ~140s/epoch
- Data: data/imagenet100/
- Running via run_resilient.sh (auto-restart on crash)

## Useful Commands
```bash
# Local resilient training
bash run_resilient.sh

# Run training (detached, Modal)
modal run --detach train_modal.py::train --model register --lr 1e-3 --epochs 400

# Check logs
modal app logs <app-id>

# Download results
modal volume get vit-checkpoints /<model>/ ./modal_output/<model>/ --force

# List volume contents
modal volume ls vit-checkpoints /<model>/
```
