"""ImageNet-100 data loading with DeiT III 3-Augment."""

import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFilter, ImageOps
from timm.data import Mixup


class ThreeAugment:
    """Randomly apply one of: grayscale, solarize, or gaussian blur."""

    def __call__(self, img):
        op = random.choice([0, 1, 2])
        if op == 0:
            return ImageOps.grayscale(img).convert('RGB')
        elif op == 1:
            return ImageOps.solarize(img, threshold=128)
        else:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ThreeAugment(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_mixup(args):
    """Create timm Mixup object for DeiT III training."""
    return Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        label_smoothing=args.label_smoothing,
        num_classes=args.num_classes,
        mode='elem',
    )


def get_dataloaders(args):
    """Build train and val dataloaders from ImageFolder structure."""
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
