"""Modal training script for cs229 ViT experiments.

Step 1 - download data once:
    modal run modal_train.py::download_data

Step 2 - smoke test:
    modal run modal_train.py -- --model register --num_registers 4 --epochs 2 --no_wandb

Step 3 - full run:
    modal run modal_train.py -- --model register --num_registers 4 --lr 3e-3 --wandb_run_name register_K4_lr3e-3
"""

import modal, os, sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "torchvision", "timm", "wandb", "matplotlib", "numpy", "pyyaml", "datasets", "Pillow"])
    .add_local_dir(".", remote_path="/cs229_project")
)

app = modal.App("vit-cs229", image=image)

data_volume   = modal.Volume.from_name("imagenet100-data", create_if_missing=True)
output_volume = modal.Volume.from_name("vit-output",       create_if_missing=True)


@app.function(volumes={"/data": data_volume}, timeout=60*60*3, image=image)
def download_data():
    from datasets import load_dataset
    target = "/data/imagenet100"
    if os.path.exists(f"{target}/train") and os.listdir(f"{target}/train"):
        print("Already downloaded."); return
    ds = load_dataset("clane9/imagenet-100", cache_dir="/data/hf_cache")
    for hf_split, folder in [("train", "train"), ("validation", "val")]:
        for idx, sample in enumerate(ds[hf_split]):
            cls = sample.get("synset", f"n{int(sample['label']):08d}")
            out = f"{target}/{folder}/{cls}"
            os.makedirs(out, exist_ok=True)
            sample["image"].convert("RGB").save(f"{out}/{idx:07d}.JPEG", format="JPEG")
        print(f"{hf_split} done.")
    data_volume.commit()


@app.function(
    gpu="A100", timeout=60*60*24,
    volumes={"/data": data_volume, "/output": output_volume},
    image=image,
)
def run(*train_args):
    import subprocess
    os.chdir("/cs229_project")
    cmd = [sys.executable, "train.py", "--data_path", "/data/imagenet100", "--output_dir", "/output"] + list(train_args)
    subprocess.run(cmd, check=True)
    output_volume.commit()


@app.local_entrypoint()
def main(*args):
    run.remote(*args)