"""
Attention Mechanism Zoo experiment on Modal.

Trains ViT-S with different attention mechanisms and normalization types
on ImageNet-100, saving layer features for CKA analysis.

Usage:
  # Standard softmax baseline
  modal run train_modal.py::train --attn-type softmax

  # Sigmoid attention (Apple ICLR 2025)
  modal run train_modal.py::train --attn-type sigmoid

  # ReLU attention (Wortsman 2023)
  modal run train_modal.py::train --attn-type relu

  # Gated SDPA (Qiu et al. NeurIPS 2025)
  modal run train_modal.py::train --attn-type gated

  # Linear attention (Katharopoulos 2020)
  modal run train_modal.py::train --attn-type linear

  # DyT normalization (He/LeCun CVPR 2025) + softmax attention
  modal run train_modal.py::train --attn-type softmax --norm-type dyt

  # Download results
  modal volume get gated-attn-checkpoints / ./results/
"""

import modal

app = modal.App("attn-zoo-experiment")

data_volume = modal.Volume.from_name("imagenet100-data", create_if_missing=True)
output_volume = modal.Volume.from_name("gated-attn-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "timm>=0.9.0",
        "matplotlib",
        "numpy",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

VOLUME_MOUNTS = {"/data": data_volume, "/output": output_volume}


def _train_impl(
    attn_type: str,
    norm_type: str,
    model_size: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    layer_scale_init: float,
    drop_path_rate: float,
    resume: bool,
    num_workers: int = 10,
    seed: int = 0,
    run_suffix: str = "",
):
    import sys, os
    sys.path.insert(0, "/root/project")

    import torch
    from types import SimpleNamespace
    from src.models import ViT
    from src.data import get_dataloaders, get_mixup
    from src.trainer import Trainer

    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {seed}")

    # ── Extract data ──
    import subprocess, time as _time, json
    tar_path = "/data/imagenet100.tar"
    data_path = "/tmp/imagenet100"

    if not os.path.isfile(tar_path):
        print("ERROR: imagenet100.tar not found. Upload first:")
        print("  modal volume put imagenet100-data /path/to/imagenet100.tar /imagenet100.tar")
        return

    print("Extracting data...")
    t_copy = _time.time()
    os.makedirs(data_path, exist_ok=True)
    subprocess.run(["tar", "xf", tar_path, "-C", data_path], check=True)
    subprocess.run(["find", data_path, "-name", "._*", "-delete"], check=True)

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

    val_src = os.path.join(data_path, "val.X")
    val_dir = os.path.join(data_path, "val")
    if os.path.isdir(val_src) and not os.path.exists(val_dir):
        os.rename(val_src, val_dir)

    print(f"Data extracted in {_time.time() - t_copy:.0f}s")

    # ── Run name ──
    parts = [attn_type]
    if norm_type != 'layernorm':
        parts.append(norm_type)
    if run_suffix:
        parts.append(run_suffix)
    run_name = "_".join(parts)
    output_dir = f"/output/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # ── Save config ──
    config = {
        'attn_type': attn_type, 'norm_type': norm_type,
        'model_size': model_size,
        'lr': lr, 'weight_decay': weight_decay,
        'layer_scale_init': layer_scale_init,
        'drop_path_rate': drop_path_rate,
        'epochs': epochs, 'batch_size': batch_size,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config: {config}")

    # Feature-saving epochs: start, mid-training, and end
    feature_epochs = [0, 10, 25, 50, 75]
    if epochs > 100:
        feature_epochs.extend([100, 150, 200, 300])
    feature_epochs = [e for e in feature_epochs if e < epochs]

    args = SimpleNamespace(
        num_classes=100,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad=1.0,
        epochs=epochs,
        warmup_epochs=5,
        warmup_lr=1e-6,
        mixup=0.8,
        cutmix=1.0,
        label_smoothing=0.1,
        amp=True,
        output_dir=output_dir,
        save_freq=25,
        analysis_freq=5,
        feature_epochs=feature_epochs,
    )

    model = ViT(
        model_size=model_size,
        num_classes=100,
        attn_type=attn_type,
        norm_type=norm_type,
        layer_scale_init=layer_scale_init,
        drop_path_rate=drop_path_rate,
    )
    num_params = model.num_params()
    print(f"Model: ViT-{model_size[0].upper()} | attn={attn_type} | "
          f"norm={norm_type} | {num_params:.1f}M params")

    train_loader, val_loader = get_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    mixup_fn = get_mixup(args)
    trainer = Trainer(model, train_loader, val_loader, mixup_fn, args)

    ckpt_path = os.path.join(output_dir, "checkpoint_latest.pth")
    if resume and os.path.exists(ckpt_path):
        trainer.resume_from_checkpoint(ckpt_path)

    trainer.train()
    output_volume.commit()
    print(f"Done! Results saved to gated-attn-checkpoints:/{run_name}/")


@app.function(
    image=image, gpu="A100", cpu=12,
    volumes=VOLUME_MOUNTS, timeout=86400,
)
def train(
    attn_type: str = "softmax",
    norm_type: str = "layernorm",
    model_size: str = "small",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    layer_scale_init: float = 1e-4,
    drop_path_rate: float = 0.1,
    resume: bool = False,
    seed: int = 0,
    run_suffix: str = "",
):
    _train_impl(
        attn_type, norm_type, model_size, epochs, batch_size,
        lr, weight_decay, layer_scale_init, drop_path_rate, resume,
        seed=seed, run_suffix=run_suffix,
    )
