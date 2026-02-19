# CS229 Project: Eliminating High-Norm Artifacts in Vision Transformers

## Project Goal
Investigate and eliminate high-norm token artifacts that emerge in Vision Transformers during training. The project trains ViT-Small on ImageNet-100 following the DeiT III protocol, then compares vanilla ViT against gated and register variants to study how architectural changes affect artifact formation.

## Models
| Model | Class | Description | Params |
|-------|-------|-------------|--------|
| vanilla | VanillaViT | timm DeiT-III Small wrapper | 21.7M |
| sdpa | SDPViT | Custom ViT with F.scaled_dot_product_attention | 21.7M |
| register | RegisterViT | SDPA + k=4 learnable register tokens | 21.7M |

## File Structure
cs229_project/
├── train.py                # Local training entry point (argparse)
├── train_modal.py          # Modal cloud training (--detach for reliability)
├── train_colab.ipynb       # Colab notebook
├── requirements.txt
├── src/
│   ├── init.py
│   ├── models.py           # VanillaViT, SDPViT, RegisterViT + get_model()
│   ├── data.py             # ThreeAugment, ImageFolder loaders, timm Mixup
│   ├── trainer.py          # DeiT III training loop, CSV logging, L2 analysis
│   └── analysis.py         # compute_l2_norms, plot_l2_distribution,
visualize_attention_maps
├── modal_output/           # Downloaded results from Modal
│   └── sdpa/               # SDPA 200-epoch run (complete)
├── archive/                # Raw ImageNet-100 (train.X1-X4, val.X)
└── context.md

## Training Recipe (DeiT III)
- Optimizer: LAMB (lr configurable, weight_decay=0.05)
- Schedule: cosine with 5 warmup epochs (warmup_lr=1e-6)
- Loss: BCE with label smoothing via timm Mixup
- Augmentation: 3-Augment + ColorJitter(0.3), Mixup=0.8, CutMix=1.0
- AMP (CUDA only), gradient clipping max_norm=1.0

## Modal Deployment
- GPU: A100-SXM4-40GB, 12 CPUs, 10 DataLoader workers
- Data: imagenet100.tar on volume "imagenet100-data", extracted to local SSD
- Output: volume "vit-checkpoints" (/{model}/ subdirs)
- Use `--detach` for runs that survive local disconnects

## Saved Outputs Per Run
- `metrics.csv`: every epoch (train_loss, val_loss, val_acc1, val_acc5, lr,
l2_norm_mean, l2_norm_std, l2_outlier_ratio, epoch_time_s)
- `l2_norms/epoch{N}.npy`: raw L2 norm arrays every epoch
- `checkpoint_epoch{N}.pth`: every 25 epochs + latest
- Plots (l2_dist, attn_maps): every 10 epochs only (can regenerate from
checkpoints/norms)

## Completed Experiments
### SDPA ViT — 200 epochs, lr=4e-3
- Final: val_acc@1=65.96%, val_acc@5=88.56%
- L2 norm: mean=8.87, std=0.38, outlier_ratio=0.0007
- Results downloaded to modal_output/sdpa/

## Running Experiments
### Register ViT — 200 epochs, lr=1e-3, k=4 register tokens
- Modal app: ap-u6fyx3itOXSM4nPCIcTdSV
- Dashboard: https://modal.com/apps/vincexxwu/main/ap-u6fyx3itOXSM4nPCIcTdSV
- Download: `modal volume get vit-checkpoints /register/
./modal_output/register/ --force`

## Useful Commands
```bash
# Run training (detached)
modal run --detach train_modal.py::train --model register --lr 1e-3 --epochs
200

# Check logs
modal app logs <app-id>

# Download results
modal volume get vit-checkpoints /<model>/ ./modal_output/<model>/ --force

# List volume contents
modal volume ls vit-checkpoints /<model>/