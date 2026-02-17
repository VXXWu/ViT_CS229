#!/bin/bash
# =============================================================================
# Training wrapper that syncs checkpoints to GCS after each save.
# Use this on spot/preemptible VMs to avoid losing progress.
#
# Usage:
#   bash gcp/run_training.sh [extra args for train.py]
#
# Examples:
#   bash gcp/run_training.sh
#   bash gcp/run_training.sh --epochs 400 --lr 2e-3
#   bash gcp/run_training.sh --model register
# =============================================================================

set -e

# ---- Detect GCS bucket ----
GCS_BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs_bucket" \
    -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
if [ -z "$GCS_BUCKET" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    GCS_BUCKET="gs://${PROJECT_ID}-vit-training"
fi

OUTPUT_DIR="./output"
DATA_PATH="./data/imagenet100"

# ---- Resume flag ----
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/checkpoint_latest.pth" ]; then
    RESUME_FLAG="--resume $OUTPUT_DIR/checkpoint_latest.pth"
    echo "Resuming from $OUTPUT_DIR/checkpoint_latest.pth"
fi

# ---- Background sync: watch for new checkpoints and upload to GCS ----
sync_checkpoints() {
    echo "[GCS Sync] Watching $OUTPUT_DIR for checkpoints..."
    LAST_SYNCED=""
    while true; do
        sleep 60
        LATEST="$OUTPUT_DIR/checkpoint_latest.pth"
        if [ -f "$LATEST" ]; then
            CURRENT_HASH=$(md5sum "$LATEST" 2>/dev/null | cut -d' ' -f1 || md5 -q "$LATEST" 2>/dev/null)
            if [ "$CURRENT_HASH" != "$LAST_SYNCED" ]; then
                echo "[GCS Sync] Uploading checkpoint..."
                gsutil -q cp "$LATEST" "$GCS_BUCKET/checkpoints/checkpoint_latest.pth"
                # Also sync any epoch-specific checkpoints
                gsutil -q -m rsync -x '.*checkpoint_latest.*' "$OUTPUT_DIR/" "$GCS_BUCKET/checkpoints/"
                LAST_SYNCED="$CURRENT_HASH"
                echo "[GCS Sync] Done."
            fi
        fi
    done
}

# Start background sync
sync_checkpoints &
SYNC_PID=$!
trap "kill $SYNC_PID 2>/dev/null; exit" EXIT INT TERM

# ---- Run training ----
echo "Starting training..."
echo "  GCS bucket: $GCS_BUCKET"
echo "  Output dir: $OUTPUT_DIR"
echo ""

python train.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 800 \
    --wandb_run_name "vanilla_800ep" \
    $RESUME_FLAG \
    "$@"

# ---- Final sync ----
echo "Training complete. Final checkpoint sync..."
gsutil -m rsync "$OUTPUT_DIR/" "$GCS_BUCKET/checkpoints/"
echo "All checkpoints synced to $GCS_BUCKET/checkpoints/"

# Sync analysis plots
gsutil -m cp "$OUTPUT_DIR"/*.png "$GCS_BUCKET/analysis/" 2>/dev/null || true
echo "Analysis plots synced to $GCS_BUCKET/analysis/"
