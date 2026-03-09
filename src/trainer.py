"""Trainer with proper WD exclusion, artifact detection, and representation saving."""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from timm.scheduler import CosineLRScheduler

CSV_COLUMNS = [
    'epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc5', 'lr',
    'l2_norm_mean', 'l2_norm_std', 'l2_outlier_ratio',
    'l2_norm_max', 'l2_max_mean_ratio',
    'gate_mean', 'gate_sparsity_01', 'gate_sparsity_09',
    'attn_entropy_mean',
    'epoch_time_s',
]


def param_groups_weight_decay(model, weight_decay):
    """Proper WD exclusion matching DeiT-III / timm convention.

    All 1D params (LayerScale gammas, LayerNorm/DyT weights) and biases
    get weight_decay=0. Only 2D+ weight matrices get WD.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


class Trainer:
    def __init__(self, model, train_loader, val_loader, mixup_fn, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mixup_fn = mixup_fn
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.model.to(self.device)

        # Proper param groups: exclude 1D params from WD
        param_groups = param_groups_weight_decay(model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.999))

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=args.epochs,
            warmup_t=args.warmup_epochs,
            warmup_lr_init=args.warmup_lr,
            lr_min=1e-6,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.use_amp = args.amp and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.start_epoch = 0

        os.makedirs(args.output_dir, exist_ok=True)
        self.csv_path = os.path.join(args.output_dir, 'metrics.csv')

    def _init_csv(self):
        if self.start_epoch > 0 and os.path.exists(self.csv_path):
            return
        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def _log_csv(self, row):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})

    def resume_from_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = ckpt['epoch'] + 1
        # Don't restore old scheduler — rebuild with new t_initial so
        # cosine schedule extends to the new total epochs.
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(scheduler rebuilt for {self.args.epochs} total epochs)")

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        path = os.path.join(self.args.output_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(state, path)
        latest = os.path.join(self.args.output_dir, 'checkpoint_latest.pth')
        torch.save(state, latest)

    def train(self):
        self._init_csv()

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            self.scheduler.step(epoch)
            train_loss = self._train_one_epoch()
            val_acc1, val_acc5, val_loss = self._validate()

            lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - t0

            log_dict = {
                'epoch': epoch,
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc1': f'{val_acc1:.2f}',
                'val_acc5': f'{val_acc5:.2f}',
                'lr': f'{lr:.6f}',
                'epoch_time_s': f'{epoch_time:.0f}',
            }

            # Analysis: L2 norms, gate stats, attention entropy, layer features
            analysis_freq = getattr(self.args, 'analysis_freq', 5)
            if epoch % analysis_freq == 0 or epoch == self.args.epochs - 1:
                analysis = self._run_analysis(epoch)
                log_dict.update(analysis)

            self._log_csv(log_dict)

            save_freq = getattr(self.args, 'save_freq', 25)
            if (epoch + 1) % save_freq == 0 or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch)

            print(
                f"Epoch {epoch}/{self.args.epochs-1} | "
                f"loss={train_loss:.4f} | "
                f"val_acc@1={val_acc1:.2f}% | "
                f"lr={lr:.6f} | "
                f"time={epoch_time:.0f}s"
                + (f" | l2_max/mean={log_dict.get('l2_max_mean_ratio', 'N/A')}"
                   if 'l2_max_mean_ratio' in log_dict else "")
                + (f" | entropy={log_dict.get('attn_entropy_mean', 'N/A')}"
                   if 'attn_entropy_mean' in log_dict else "")
            )

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()
            with autocast(self.amp_device, enabled=self.use_amp):
                loss = self.criterion(self.model(images), targets)

            self.scaler.scale(loss).backward()
            clip_grad = getattr(self.args, 'clip_grad', 1.0)
            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        correct1 = correct5 = total = 0
        criterion_val = nn.CrossEntropyLoss()

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(images)
            total_loss += criterion_val(outputs, targets).item() * images.size(0)
            _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_mask = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct1 += correct_mask[:, :1].sum().item()
            correct5 += correct_mask[:, :5].sum().item()
            total += images.size(0)

        return (100.0 * correct1 / max(total, 1),
                100.0 * correct5 / max(total, 1),
                total_loss / max(total, 1))

    @torch.no_grad()
    def _run_analysis(self, epoch):
        """L2 norms, attention entropy, gate stats, and layer features for CKA."""
        self.model.eval()
        images, _ = next(iter(self.val_loader))
        images = images.to(self.device)

        # L2 norms of patch tokens
        patch_tokens = self.model.get_patch_tokens(images)
        norms = torch.norm(patch_tokens, dim=-1).cpu().numpy().flatten()
        mean_norm = float(norms.mean())
        std_norm = float(norms.std())
        max_norm = float(norms.max())
        threshold = mean_norm + 3 * std_norm
        outlier_ratio = float((norms > threshold).sum() / len(norms))
        max_mean_ratio = max_norm / max(mean_norm, 1e-8)

        norms_dir = os.path.join(self.args.output_dir, 'l2_norms')
        os.makedirs(norms_dir, exist_ok=True)
        np.save(os.path.join(norms_dir, f'epoch{epoch}.npy'), norms)

        result = {
            'l2_norm_mean': f'{mean_norm:.2f}',
            'l2_norm_std': f'{std_norm:.2f}',
            'l2_outlier_ratio': f'{outlier_ratio:.4f}',
            'l2_norm_max': f'{max_norm:.2f}',
            'l2_max_mean_ratio': f'{max_mean_ratio:.2f}',
        }

        if outlier_ratio > 0.01 or max_mean_ratio > 3.0:
            print(f"  [ARTIFACT ALERT] outlier_ratio={outlier_ratio:.4f}, "
                  f"max/mean={max_mean_ratio:.2f}")

        # Attention entropy
        entropies = self.model.get_attention_entropy(images)
        if entropies:
            all_ent = torch.stack(entropies)  # (layers, heads)
            mean_ent = float(all_ent.mean())
            result['attn_entropy_mean'] = f'{mean_ent:.4f}'

            entropy_dir = os.path.join(self.args.output_dir, 'attn_entropy')
            os.makedirs(entropy_dir, exist_ok=True)
            entropy_data = []
            for i, ent in enumerate(entropies):
                entropy_data.append({
                    'layer': i,
                    'per_head': ent.tolist(),
                    'mean': float(ent.mean()),
                })
            with open(os.path.join(entropy_dir, f'epoch{epoch}.json'), 'w') as f:
                json.dump(entropy_data, f, indent=2)

        # Gate statistics (if gated model)
        gate_scores = self.model.get_gate_scores(images)
        if gate_scores:
            all_gates = torch.cat([g.flatten() for g in gate_scores])
            gate_mean = float(all_gates.mean())
            sparsity_01 = float((all_gates < 0.1).float().mean())
            sparsity_09 = float((all_gates > 0.9).float().mean())
            result['gate_mean'] = f'{gate_mean:.4f}'
            result['gate_sparsity_01'] = f'{sparsity_01:.4f}'
            result['gate_sparsity_09'] = f'{sparsity_09:.4f}'

            gate_dir = os.path.join(self.args.output_dir, 'gate_stats')
            os.makedirs(gate_dir, exist_ok=True)
            layer_stats = []
            for i, g in enumerate(gate_scores):
                g_flat = g.flatten()
                layer_stats.append({
                    'layer': i,
                    'mean': float(g_flat.mean()),
                    'std': float(g_flat.std()),
                    'min': float(g_flat.min()),
                    'max': float(g_flat.max()),
                    'frac_below_0.1': float((g_flat < 0.1).float().mean()),
                    'frac_above_0.9': float((g_flat > 0.9).float().mean()),
                })
            with open(os.path.join(gate_dir, f'epoch{epoch}.json'), 'w') as f:
                json.dump(layer_stats, f, indent=2)

            print(f"  [Gates] mean={gate_mean:.4f}, "
                  f"<0.1={sparsity_01:.4f}, >0.9={sparsity_09:.4f}")

        # Save layer features for CKA (at key epochs)
        feature_epochs = getattr(self.args, 'feature_epochs', [])
        if epoch in feature_epochs or epoch == self.args.epochs - 1:
            self._save_layer_features(epoch)

        return result

    @torch.no_grad()
    def _save_layer_features(self, epoch):
        """Save CLS and mean-patch features at every layer over val set for CKA."""
        self.model.eval()
        all_cls = [[] for _ in range(self.model.depth + 1)]
        all_patch = [[] for _ in range(self.model.depth + 1)]

        for images, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            cls_feats, patch_feats = self.model.get_layer_features(images)
            for i, (c, p) in enumerate(zip(cls_feats, patch_feats)):
                all_cls[i].append(c.cpu())
                all_patch[i].append(p.cpu())

        cls_tensor = [torch.cat(feats, dim=0) for feats in all_cls]
        patch_tensor = [torch.cat(feats, dim=0) for feats in all_patch]

        feat_dir = os.path.join(self.args.output_dir, 'features')
        os.makedirs(feat_dir, exist_ok=True)
        torch.save({
            'cls': cls_tensor,     # list of (N_val, D) tensors
            'patch': patch_tensor, # list of (N_val, D) tensors
            'epoch': epoch,
        }, os.path.join(feat_dir, f'epoch{epoch}.pt'))
        print(f"  [Features] saved {len(cls_tensor)} layers, "
              f"{cls_tensor[0].shape[0]} samples")
