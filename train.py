"""Main training script for ViT experiments (DeiT III protocol)."""

import argparse
import os

import torch
import wandb

from src.models import get_model
from src.data import get_dataloaders, get_mixup
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='ViT Training - DeiT III Protocol')

    # Model
    parser.add_argument('--model', type=str, default='vanilla',
                        choices=['vanilla', 'register', 'gated'],
                        help='Model variant')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--drop_path', type=float, default=0.05)

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet-100 root (with train/ and val/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    # Optimizer (DeiT III: LAMB)
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad', type=float, default=1.0)

    # Schedule
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)

    # Augmentation / Regularization
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Training
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--no_amp', action='store_true')

    # Logging / Saving
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cs229-vit-artifacts')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--analysis_freq', type=int, default=10,
                        help='Run L2 norm analysis every N epochs')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Handle flag overrides
    if args.no_amp:
        args.amp = False
    if args.no_wandb:
        args.wandb = False

    return args


def main():
    args = parse_args()

    # Print config
    print("=" * 60)
    print("ViT Training - DeiT III Protocol")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    # Build model
    model = get_model(args)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model} | Parameters: {num_params:.1f}M")

    # Build dataloaders
    train_loader, val_loader = get_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # Mixup/CutMix
    mixup_fn = get_mixup(args)

    # WandB
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.model}_ep{args.epochs}",
            config=vars(args),
        )

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, mixup_fn, args)

    # Resume
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Train
    trainer.train()

    if args.wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()
