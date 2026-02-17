# CS229 Project: Eliminating High-Norm Artifacts in Vision Transformers

## Project Goal
Investigate and eliminate high-norm token artifacts that emerge in Vision Transformers during training. The project trains ViT-Small on ImageNet-100 following the DeiT III protocol, then compares vanilla ViT against gated and register variants to study how architectural changes affect artifact formation.

## Repository Structure
```
cs229_project/
├── train.py                  # Single entry point — argparse with all DeiT III defaults
├── requirements.txt          # torch, torchvision, timm, wandb, matplotlib, numpy, pyyaml
├── context.md                # This file
├── src/
│   ├── __init__.py
│   ├── models.py             # Model definitions (vanilla now; gated/register later)
│   ├── data.py               # ImageNet-100 loader + 3-Augment + Mixup/CutMix
│   ├── trainer.py            # DeiT III training loop
│   └── analysis.py           # L2 norm tracking + attention map visualization
├── output/                   # Checkpoints, L2 histograms, attention maps (gitignored)
├── data/                     # Dataset directory (gitignored)
└── venv/                     # Python virtual environment
```

## Current Status
- **Implemented**: Vanilla ViT-Small (DeiT-III backbone, ~21.7M params)
- **Verified**: End-to-end training on dummy ImageNet-100 (2 epochs, CPU/MPS)
- **Pending**: Register ViT variant, Gated ViT variant (add to `src/models.py` + `get_model()`)

## Training Recipe (DeiT III)

| Setting            | Value                              |
|--------------------|------------------------------------|
| Architecture       | `deit3_small_patch16_224` (timm)   |
| Params             | ~21.7M                             |
| Dataset            | ImageNet-100 (100 classes)         |
| Epochs             | 800                                |
| Optimizer          | LAMB (lr=4e-3, wd=0.05, betas=(0.9, 0.999)) |
| LR Schedule        | Cosine, warmup=5 epochs, warmup_lr=1e-6, min_lr=1e-6 |
| Loss               | BCEWithLogitsLoss (soft labels from Mixup) |
| Augmentation       | 3-Augment (grayscale/solarize/blur) + ColorJitter(0.3) |
| Mixup / CutMix     | alpha=0.8 / alpha=1.0, mode='elem' |
| Label Smoothing    | 0.1                                |
| Drop Path          | 0.05                               |
| Gradient Clipping  | max_norm=1.0                       |
| Mixed Precision    | AMP (CUDA only; disabled on MPS/CPU) |
| Eval Crop Ratio    | 1.0 (Resize(224) → CenterCrop(224)) |

## Module Details

### `src/models.py`
- **`VanillaViT`**: Wraps `timm.create_model('deit3_small_patch16_224')`.
  - `forward(x)` — standard classification forward pass
  - `get_patch_tokens(x)` — returns final-layer patch tokens (excludes CLS), shape `(B, 196, 384)`
  - `get_attention_maps(x)` — returns list of 12 attention tensors `(B, 6, 197, 197)` via forward hooks on each block's `Attention` module. Recomputes Q·K^T from the `qkv` projection to get raw attention weights.
- **`get_model(args)`**: Factory function. Currently supports `vanilla`; extend for `register`/`gated`.

### `src/data.py`
- **`ThreeAugment`**: Randomly picks one of grayscale, solarize (threshold=128), or Gaussian blur (radius 0.1–2.0).
- **`get_train_transform()`**: RandomResizedCrop(224) → HFlip → ThreeAugment → ColorJitter(0.3) → ToTensor → Normalize (ImageNet stats).
- **`get_val_transform()`**: Resize(224) → CenterCrop(224) → ToTensor → Normalize. Crop ratio = 1.0 per DeiT III.
- **`get_mixup(args)`**: Returns `timm.data.Mixup` with `mode='elem'` (element-wise mixing). Applied in training loop, not in transform pipeline.
- **`get_dataloaders(args)`**: Builds `ImageFolder`-based train/val `DataLoader`s with `pin_memory=True`, `drop_last=True` (train).

### `src/trainer.py`
- **`Trainer`** class handles the full training lifecycle:
  - Device auto-detection: CUDA → MPS → CPU
  - AMP only enabled on CUDA (GradScaler + autocast)
  - `_train_one_epoch()`: mixup → forward → BCE loss → grad clip → step
  - `_validate()`: CrossEntropyLoss on hard labels, computes top-1/top-5 accuracy
  - `_run_analysis()`: calls `analysis.py` functions, saves plots, logs to WandB
  - Checkpointing: every `save_freq` epochs (default 50) + final epoch + `checkpoint_latest.pth`
  - Resume: loads model, optimizer, scheduler, and scaler state from checkpoint

### `src/analysis.py`
- **`compute_l2_norms(model, images)`**: Extracts patch tokens, computes per-token L2 norms, returns norms array + outlier stats (outlier = norm > mean + 3σ).
- **`plot_l2_distribution(norms, epoch)`**: Histogram with mean and mean+3σ lines.
- **`visualize_attention_maps(model, images, epoch)`**: Grid of CLS→patch attention heatmaps (rows=images, cols=layers). Head-averaged, reshaped to 14×14.

### `train.py`
- Single `main()` entry point: parse args → build model → dataloaders → mixup → WandB init → Trainer → train loop.
- All DeiT III defaults baked into argparse defaults. Key flags:
  - `--model {vanilla,register,gated}` — model selection
  - `--data_path` (required) — ImageNet-100 root with `train/` and `val/` subdirs
  - `--resume` — path to checkpoint for resumption
  - `--no_wandb` / `--no_amp` — disable WandB logging or AMP

## Usage

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Quick test (2 epochs, no WandB)
python3 train.py --data_path ./data/imagenet100_dummy --epochs 2 --batch_size 32 --num_workers 0 --no_wandb

# Full training (800 epochs)
python3 train.py --data_path /path/to/imagenet100

# Resume from checkpoint
python3 train.py --data_path /path/to/imagenet100 --resume ./output/checkpoint_latest.pth
```

## Extending with New Model Variants

To add a new variant (e.g., register ViT):

1. Add a new class in `src/models.py` that implements `forward()`, `get_patch_tokens()`, and `get_attention_maps()`.
2. Register it in `get_model(args)` under its name.
3. It will automatically work with the existing training loop, analysis pipeline, and CLI (`--model register`).

## Key Metrics Tracked
- **Training**: `train_loss` (BCE)
- **Validation**: `val_loss` (CE), `val_acc1`, `val_acc5`
- **Analysis** (every `analysis_freq` epochs): `l2_norm_mean`, `l2_norm_std`, `l2_outlier_ratio`
- **Visualizations**: L2 norm histograms, CLS→patch attention heatmaps per layer
- **Schedule**: `lr` (logged each epoch)

## Data Format
ImageNet-100 must be organized as:
```
imagenet100/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ... (100 class folders)
└── val/
    ├── n01440764/
    │   └── ...
    └── ... (100 class folders)
```
