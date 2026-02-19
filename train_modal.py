"""
Run ViT training on Modal (https://modal.com).

Usage:
  pip install modal && modal setup

  # Upload ImageNet-100 tar to a Modal volume (one-time)
  modal volume put imagenet100-data /path/to/imagenet100.tar /imagenet100.tar

  # Train SDPA ViT for 200 epochs on A100
  modal run train_modal.py::train --model sdpa --epochs 200

  # Resume to 800 epochs
  modal run train_modal.py::train --model sdpa --epochs 800 --resume

  # Download results
  modal volume get vit-checkpoints /sdpa/ ./modal_output/
"""

import modal

app = modal.App("cs229-vit-training")

# Persistent volumes
data_volume = modal.Volume.from_name("imagenet100-data", create_if_missing=True)
output_volume = modal.Volume.from_name("vit-checkpoints", create_if_missing=True)

# Image with deps + source code baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "timm>=0.9.0",
        "wandb",
        "matplotlib",
        "numpy",
        "pyyaml",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)


@app.function(
    image=image,
    gpu="A100",
    cpu=12,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
    timeout=86400,
)
def train(
    model: str = "sdpa",
    epochs: int = 800,
    batch_size: int = 256,
    lr: float = 4e-3,
    resume: bool = False,
):
    import sys
    import os
    sys.path.insert(0, "/root/project")

    import torch
    from types import SimpleNamespace
    from src.models import get_model
    from src.data import get_dataloaders, get_mixup
    from src.trainer import Trainer

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Extract tar from volume to local SSD for fast I/O
    import subprocess
    import time as _time

    tar_path = "/data/imagenet100.tar"
    data_path = "/tmp/imagenet100"

    if not os.path.isfile(tar_path):
        print("ERROR: imagenet100.tar not found in volume.")
        print("Upload first: modal volume put imagenet100-data /path/to/imagenet100.tar /imagenet100.tar")
        return

    print("Extracting data to local SSD...")
    t_copy = _time.time()
    os.makedirs(data_path, exist_ok=True)
    subprocess.run(["tar", "xf", tar_path, "-C", data_path], check=True)

    # Remove macOS resource fork files (._*)
    subprocess.run(["find", data_path, "-name", "._*", "-delete"], check=True)

    # Merge split train dirs (train.X1-X4) into unified train/
    train_dir = os.path.join(data_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    for split in ["train.X1", "train.X2", "train.X3", "train.X4"]:
        split_path = os.path.join(data_path, split)
        if os.path.isdir(split_path):
            for cls in os.listdir(split_path):
                src = os.path.join(split_path, cls)
                dst = os.path.join(train_dir, cls)
                if os.path.isdir(src) and not os.path.exists(dst):
                    os.rename(src, dst)

    # Rename val.X to val/
    val_src = os.path.join(data_path, "val.X")
    val_dir = os.path.join(data_path, "val")
    if os.path.isdir(val_src) and not os.path.exists(val_dir):
        os.rename(val_src, val_dir)

    print(f"Data extracted in {_time.time() - t_copy:.0f}s")

    train_classes = os.listdir(train_dir)
    print(f"Data: {len(train_classes)} train classes")

    output_dir = f"/output/{model}"
    os.makedirs(output_dir, exist_ok=True)

    args = SimpleNamespace(
        model=model,
        num_classes=100,
        drop_path=0.05,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        lr=lr,
        weight_decay=0.05,
        clip_grad=1.0,
        epochs=epochs,
        warmup_epochs=5,
        warmup_lr=1e-6,
        mixup=0.8,
        cutmix=1.0,
        label_smoothing=0.1,
        amp=True,
        output_dir=output_dir,
        wandb=False,
        save_freq=25,
        analysis_freq=1,
    )

    mdl = get_model(args)
    num_params = sum(p.numel() for p in mdl.parameters()) / 1e6
    print(f"Model: {model} | Params: {num_params:.1f}M")

    train_loader, val_loader = get_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    mixup_fn = get_mixup(args)
    trainer = Trainer(mdl, train_loader, val_loader, mixup_fn, args)

    ckpt_path = os.path.join(output_dir, "checkpoint_latest.pth")
    if resume and os.path.exists(ckpt_path):
        trainer.resume_from_checkpoint(ckpt_path)

    trainer.train()

    output_volume.commit()
    print("Training complete! Checkpoints saved to volume 'vit-checkpoints'.")
