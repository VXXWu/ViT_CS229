#!/bin/bash
# =============================================================================
# VM Startup Script
# Run this ON the GCP VM after SSHing in.
#
# Usage:
#   cd ~/cs229_project
#   bash gcp/startup.sh
# =============================================================================

set -e

# ---- Detect GCS bucket from VM metadata or use default ----
GCS_BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs_bucket" \
    -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
if [ -z "$GCS_BUCKET" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    GCS_BUCKET="gs://${PROJECT_ID}-vit-training"
fi
echo "Using GCS bucket: $GCS_BUCKET"

# ---- Step 1: Install dependencies ----
echo "[1/4] Installing Python dependencies..."
pip install -q -r requirements.txt

# ---- Step 2: Download data from GCS ----
echo "[2/4] Downloading ImageNet-100 from GCS..."
mkdir -p data
if ! [ -d "data/imagenet100/train" ]; then
    gsutil -m cp -r "$GCS_BUCKET/data/imagenet100" data/ || {
        echo "ERROR: No data found at $GCS_BUCKET/data/imagenet100"
        echo "Upload your data first: gsutil -m cp -r /path/to/imagenet100 $GCS_BUCKET/data/"
        exit 1
    }
else
    echo "  Data already present, skipping download"
fi

# ---- Step 3: Resume from latest checkpoint if available ----
echo "[3/4] Checking for existing checkpoints..."
mkdir -p output
gsutil cp "$GCS_BUCKET/checkpoints/checkpoint_latest.pth" output/checkpoint_latest.pth 2>/dev/null && \
    echo "  Resumed checkpoint found" || \
    echo "  No previous checkpoint, starting fresh"

# ---- Step 4: Setup WandB ----
echo "[4/4] WandB setup..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  Set your WandB API key:"
    echo "    export WANDB_API_KEY=your_key_here"
    echo "  Or run: wandb login"
    echo "  Or use --no_wandb to skip"
fi

# ---- Print GPU info ----
echo ""
echo "============================================"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU detected"
echo "============================================"

# ---- Build the training command ----
RESUME_FLAG=""
if [ -f "output/checkpoint_latest.pth" ]; then
    RESUME_FLAG="--resume output/checkpoint_latest.pth"
fi

echo ""
echo "Ready! Run training with:"
echo ""
echo "  python train.py \\"
echo "    --data_path ./data/imagenet100 \\"
echo "    --epochs 800 \\"
echo "    --output_dir ./output \\"
echo "    --wandb_run_name vanilla_800ep \\"
echo "    $RESUME_FLAG"
echo ""
echo "Or with the sync wrapper (recommended for spot VMs):"
echo ""
echo "  bash gcp/run_training.sh"
echo ""
