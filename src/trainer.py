"""DeiT III training loop with persistent CSV metrics logging."""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler

import wandb

from .analysis import compute_l2_norms, plot_l2_distribution

# All metric columns logged to CSV every epoch
CSV_COLUMNS = [
    'epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc5', 'lr',
    'l2_norm_mean', 'l2_norm_std', 'l2_outlier_ratio', 'epoch_time_s',
]


class Trainer:
    def __init__(self, model, train_loader, val_loader, mixup_fn, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mixup_fn = mixup_fn
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'

        self.model.to(self.device)

        # LAMB optimizer (DeiT III)
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine schedule with warmup
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=args.epochs,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.warmup_lr,
            lr_min=1e-6,
        )

        # BCE loss (DeiT III uses BCE with mixup-produced soft labels)
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision
        self.use_amp = args.amp and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.start_epoch = 0

        # CSV metrics log
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.args.output_dir, 'metrics.csv')

    def _init_csv(self):
        """Initialize CSV file — append if resuming, create if fresh."""
        if self.start_epoch > 0 and os.path.exists(self.csv_path):
            return  # resuming, CSV already has prior rows
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

    def _log_csv(self, row):
        """Append one row to the CSV metrics log."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})

    def resume_from_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    def save_checkpoint(self, epoch):
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, f'checkpoint_epoch{epoch}.pth')
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(state, path)
        # Also save as latest (for easy resume)
        latest = os.path.join(self.args.output_dir, 'checkpoint_latest.pth')
        torch.save(state, latest)
        print(f"Saved checkpoint: {path}")

    def train(self):
        self._init_csv()

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            self.scheduler.step(epoch)
            train_loss = self._train_one_epoch(epoch)
            val_acc1, val_acc5, val_loss = self._validate(epoch)

            lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - t0

            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc5': val_acc5,
                'lr': lr,
                'epoch_time_s': round(epoch_time, 1),
            }

            # L2 norm analysis every N epochs
            if (epoch + 1) % self.args.analysis_freq == 0 or epoch == 0:
                norms_dict = self._run_analysis(epoch)
                log_dict.update(norms_dict)

            # Log to CSV (every epoch, persists to disk immediately)
            self._log_csv(log_dict)

            if self.args.wandb:
                wandb.log(log_dict, step=epoch)

            # Checkpointing: every save_freq + always save latest
            # On Colab/GCP, save_freq=25 is safer than 50 since sessions can die
            if (epoch + 1) % self.args.save_freq == 0 or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch)

            print(
                f"Epoch {epoch}/{self.args.epochs-1} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc@1={val_acc1:.2f}% | "
                f"val_acc@5={val_acc5:.2f}% | "
                f"lr={lr:.6f} | "
                f"time={epoch_time:.0f}s"
            )

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # CHANGE THE I >= 20 AFTERWARDS
        # for images, targets in self.train_loader:
        for i, (images, targets) in enumerate(self.train_loader):
            #if i >= 2:#REMOVE AFTER
            #    break #REMOVE AFTER
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply mixup/cutmix
            images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()

            with autocast(self.amp_device, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.args.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct1 = 0
        correct5 = 0
        total = 0

        criterion_val = nn.CrossEntropyLoss()

        # CHANGE THIS BACK TOO
        for i, (images, targets) in enumerate(self.val_loader):
            #if i >= 2:
            #    break
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = criterion_val(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Top-1 and Top-5 accuracy
            _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_mask = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct1 += correct_mask[:, :1].sum().item()
            correct5 += correct_mask[:, :5].sum().item()
            total += images.size(0)

        acc1 = 100.0 * correct1 / max(total, 1)
        acc5 = 100.0 * correct5 / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return acc1, acc5, avg_loss

    @torch.no_grad()
    def _run_analysis(self, epoch):
        """Run L2 norm analysis — saves raw norms to .npy every epoch."""
        self.model.eval()
        import numpy as np

        # Grab a batch from validation set
        images, _ = next(iter(self.val_loader))
        images = images.to(self.device)

        norms, outlier_ratio, mean_norm, std_norm = compute_l2_norms(self.model, images)

        log_dict = {
            'l2_norm_mean': mean_norm,
            'l2_norm_std': std_norm,
            'l2_outlier_ratio': outlier_ratio,
        }

        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save raw L2 norms for custom figure generation later
        norms_dir = os.path.join(self.args.output_dir, 'l2_norms')
        os.makedirs(norms_dir, exist_ok=True)
        np.save(os.path.join(norms_dir, f'epoch{epoch}.npy'), norms)

        print(f"  [Analysis] L2 mean={mean_norm:.2f}, std={std_norm:.2f}, outlier_ratio={outlier_ratio:.4f}")
        return log_dict
