#!/bin/bash
# =============================================================================
# GCP VM Setup Script
# Run this from your LOCAL machine to create and configure a GPU VM.
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. gcloud auth login
#   3. gcloud config set project YOUR_PROJECT_ID
#   4. Enable Compute Engine API
#   5. Request GPU quota in your region (IAM & Admin > Quotas)
#
# Usage:
#   chmod +x gcp/setup_vm.sh
#   ./gcp/setup_vm.sh
# =============================================================================

set -e

# ---- Configuration (edit these) ----
PROJECT_ID=$(gcloud config get-value project)
VM_NAME="vit-training"
ZONE="us-central1-a"
# A100 40GB: fastest for training, a2-highgpu-1g includes 1x A100 + 12 vCPUs + 85GB RAM
MACHINE_TYPE="a2-highgpu-1g"
GPU_TYPE="nvidia-tesla-a100"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
# Set to true for ~60-70% cost savings (VM can be preempted)
USE_SPOT=true
# GCS bucket for data and checkpoints
GCS_BUCKET="gs://${PROJECT_ID}-vit-training"

echo "============================================"
echo "GCP VM Setup for ViT Training"
echo "============================================"
echo "  Project:  $PROJECT_ID"
echo "  VM:       $VM_NAME"
echo "  Zone:     $ZONE"
echo "  GPU:      ${GPU_COUNT}x $GPU_TYPE"
echo "  Spot:     $USE_SPOT"
echo "  Bucket:   $GCS_BUCKET"
echo "============================================"

# ---- Step 1: Create GCS bucket ----
echo "[1/4] Creating GCS bucket..."
gsutil mb -l us-central1 "$GCS_BUCKET" 2>/dev/null || echo "  Bucket already exists"

# ---- Step 2: Upload project code ----
echo "[2/4] Uploading project code..."
# Exclude venv, data, output, __pycache__
tar czf /tmp/cs229_project.tar.gz \
    --exclude='venv' \
    --exclude='data' \
    --exclude='output' \
    --exclude='__pycache__' \
    --exclude='.claude' \
    -C "$(dirname "$0")/.." .
gsutil cp /tmp/cs229_project.tar.gz "$GCS_BUCKET/code/cs229_project.tar.gz"
rm /tmp/cs229_project.tar.gz

# ---- Step 3: Upload ImageNet-100 data (if local) ----
echo "[3/4] Checking for ImageNet-100 data..."
if [ -d "data/imagenet100" ]; then
    echo "  Uploading ImageNet-100 to GCS (this may take a while)..."
    gsutil -m cp -r data/imagenet100 "$GCS_BUCKET/data/"
else
    echo "  No local data/imagenet100 found."
    echo "  Upload manually: gsutil -m cp -r /path/to/imagenet100 $GCS_BUCKET/data/"
fi

# ---- Step 4: Create VM ----
echo "[4/4] Creating VM..."
SPOT_FLAG=""
if [ "$USE_SPOT" = true ]; then
    SPOT_FLAG="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --maintenance-policy=TERMINATE \
    --scopes="storage-rw" \
    --metadata="gcs_bucket=$GCS_BUCKET" \
    $SPOT_FLAG

echo ""
echo "============================================"
echo "VM created! Next steps:"
echo "============================================"
echo ""
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. Run the startup script on the VM:"
echo "   gsutil cp $GCS_BUCKET/code/cs229_project.tar.gz . && bash gcp/startup.sh"
echo ""
echo "   OR copy-paste these commands:"
echo "   gsutil cp $GCS_BUCKET/code/cs229_project.tar.gz /tmp/"
echo "   mkdir -p ~/cs229_project && cd ~/cs229_project"
echo "   tar xzf /tmp/cs229_project.tar.gz"
echo "   bash gcp/startup.sh"
echo ""
echo "3. To stop the VM when done (to stop billing):"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "4. To delete the VM:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
