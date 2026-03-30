#!/usr/bin/env bash
# Deploy InftyThink to a GCP GPU VM and run training.
#
# Usage:
#   bash scripts/gcp_deploy.sh              # create VM + upload + train
#   bash scripts/gcp_deploy.sh --no-create  # skip VM creation (already exists)
#   bash scripts/gcp_deploy.sh --download   # download results from VM
#   bash scripts/gcp_deploy.sh --delete     # delete the VM
#
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────
PROJECT="creditlab-491502"
INSTANCE="inftythink-gpu"
MACHINE_TYPE="g2-standard-8"           # 8 vCPUs, 32GB RAM (required for L4)
GPU_TYPE="nvidia-l4"                   # L4 24GB — good availability, plenty for 35M params
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

# Zones to try in order (T4 availability varies)
ZONES=("us-west1-a" "us-west1-b" "us-west4-a" "us-west4-b"
       "us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f"
       "us-east1-c" "us-east1-d"
       "europe-west1-b" "europe-west1-c" "europe-west4-a" "europe-west4-b")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ─────────────────────────────────────────────────────────────────────

CMD="${1:-}"

# Helper: find the zone of an existing instance
find_zone() {
    gcloud compute instances list \
        --project="$PROJECT" \
        --filter="name=$INSTANCE" \
        --format="value(zone)" 2>/dev/null | head -1
}

# ── Download results ──
if [[ "$CMD" == "--download" ]]; then
    ZONE=$(find_zone)
    if [[ -z "$ZONE" ]]; then
        echo "ERROR: Instance $INSTANCE not found"
        exit 1
    fi
    echo "=== Downloading results from $INSTANCE (zone: $ZONE) ==="
    gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "$INSTANCE":~/reasoning/results/ "$REPO_ROOT/results/"
    gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
        "$INSTANCE":~/reasoning/checkpoints/ "$REPO_ROOT/checkpoints/"
    echo "Downloaded to $REPO_ROOT/results/ and $REPO_ROOT/checkpoints/"
    exit 0
fi

# ── Delete VM ──
if [[ "$CMD" == "--delete" ]]; then
    ZONE=$(find_zone)
    if [[ -z "$ZONE" ]]; then
        echo "Instance $INSTANCE not found — nothing to delete."
        exit 0
    fi
    echo "=== Deleting $INSTANCE (zone: $ZONE) ==="
    gcloud compute instances delete "$INSTANCE" \
        --zone="$ZONE" --project="$PROJECT" --quiet
    exit 0
fi

# ── Create VM (try zones until one works) ──
ZONE=""
if [[ "$CMD" != "--no-create" ]]; then
    echo "=== Creating GPU VM: $INSTANCE ==="
    echo "    Machine: $MACHINE_TYPE + $GPU_COUNT x $GPU_TYPE"
    echo "    Trying zones: ${ZONES[*]}"

    for z in "${ZONES[@]}"; do
        echo -n "  Trying $z ... "
        if gcloud compute instances create "$INSTANCE" \
            --project="$PROJECT" \
            --zone="$z" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
            --maintenance-policy=TERMINATE \
            --boot-disk-size="$BOOT_DISK_SIZE" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --metadata="install-nvidia-driver=True" \
            --scopes="default,storage-rw" 2>/dev/null; then
            ZONE="$z"
            echo "SUCCESS"
            break
        else
            echo "unavailable"
        fi
    done

    if [[ -z "$ZONE" ]]; then
        echo "ERROR: Could not create VM in any zone. All T4s exhausted."
        echo "Try again later or switch GPU_TYPE to nvidia-l4 in this script."
        exit 1
    fi

    echo "VM created in $ZONE. Waiting 60s for boot..."
    sleep 60
else
    ZONE=$(find_zone)
    if [[ -z "$ZONE" ]]; then
        echo "ERROR: Instance $INSTANCE not found. Run without --no-create first."
        exit 1
    fi
    echo "Using existing instance in zone: $ZONE"
fi

# ── Upload code (exclude .venv, checkpoints, .git, __pycache__) ──
echo "=== Uploading code to $INSTANCE ==="
cd "$REPO_ROOT"

# Create a tarball excluding heavy dirs, upload, and extract
tar czf /tmp/inftythink-code.tar.gz \
    --exclude='.venv' \
    --exclude='checkpoints' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='results/figures' \
    -C "$REPO_ROOT/.." "$(basename "$REPO_ROOT")"

gcloud compute scp /tmp/inftythink-code.tar.gz "$INSTANCE":~ \
    --zone="$ZONE" --project="$PROJECT" \
    -- -o StrictHostKeyChecking=no

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="rm -rf ~/reasoning && tar xzf ~/inftythink-code.tar.gz -C ~ && mv ~/reasoning ~/reasoning 2>/dev/null || true"

rm /tmp/inftythink-code.tar.gz

# ── Setup VM environment ──
echo "=== Setting up VM environment ==="
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="chmod +x ~/reasoning/scripts/gcp_setup_vm.sh && bash ~/reasoning/scripts/gcp_setup_vm.sh"

# ── Install tmux if missing ──
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="which tmux || sudo apt-get install -y tmux"

# ── Run training in tmux (persists after SSH disconnect) ──
echo "=== Starting training in tmux session 'train' ==="
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="chmod +x ~/reasoning/scripts/gcp_train.sh && tmux new-session -d -s train 'bash ~/reasoning/scripts/gcp_train.sh 2>&1 | tee ~/reasoning/train.log'"

echo ""
echo "============================================"
echo " VM is training in zone: $ZONE"
echo ""
echo " To monitor:"
echo "   gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT"
echo "   tmux attach -t train"
echo ""
echo " To download results when done:"
echo "   bash scripts/gcp_deploy.sh --download"
echo ""
echo " To delete VM when done:"
echo "   bash scripts/gcp_deploy.sh --delete"
echo "============================================"
