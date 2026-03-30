#!/usr/bin/env bash
# Deploy InftyThink to an Azure GPU VM and run training.
#
# Usage:
#   bash scripts/azure_deploy.sh              # create VM + upload + train
#   bash scripts/azure_deploy.sh --no-create  # skip VM creation (already exists)
#   bash scripts/azure_deploy.sh --download   # download results from VM
#   bash scripts/azure_deploy.sh --delete     # delete the resource group + VM
#   bash scripts/azure_deploy.sh --ssh        # SSH into the VM
#
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────
SUBSCRIPTION="49478ef1-9886-44f3-90ce-2a120dc96b3e"
RESOURCE_GROUP="inftythink-rg"
VM_NAME="inftythink-gpu"
LOCATION="westus2"
VM_SIZE="Standard_NC4as_T4_v3"        # 4 vCPUs, 28GB RAM, 1x T4 16GB — ~$0.53/hr
ADMIN_USER="azureuser"
IMAGE="Canonical:ubuntu-24_04-lts:server:latest"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ─────────────────────────────────────────────────────────────────────

CMD="${1:-}"

# Helper: get VM public IP
get_ip() {
    az vm show -g "$RESOURCE_GROUP" -n "$VM_NAME" -d --query publicIps -o tsv 2>/dev/null
}

# ── SSH ──
if [[ "$CMD" == "--ssh" ]]; then
    IP=$(get_ip)
    echo "SSH into $VM_NAME at $IP"
    ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$IP"
    exit 0
fi

# ── Download results ──
if [[ "$CMD" == "--download" ]]; then
    IP=$(get_ip)
    echo "=== Downloading results from $VM_NAME ($IP) ==="
    scp -o StrictHostKeyChecking=no -r "$ADMIN_USER@$IP":~/reasoning/results/ "$REPO_ROOT/results/"
    scp -o StrictHostKeyChecking=no -r "$ADMIN_USER@$IP":~/reasoning/checkpoints/ "$REPO_ROOT/checkpoints/"
    echo "Downloaded to $REPO_ROOT/results/ and $REPO_ROOT/checkpoints/"
    exit 0
fi

# ── Delete everything ──
if [[ "$CMD" == "--delete" ]]; then
    echo "=== Deleting resource group $RESOURCE_GROUP (VM + all resources) ==="
    az group delete -n "$RESOURCE_GROUP" --yes --no-wait
    echo "Deletion started (runs in background)."
    exit 0
fi

# ── Create VM ──
if [[ "$CMD" != "--no-create" ]]; then
    echo "=== Creating resource group: $RESOURCE_GROUP ==="
    az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

    echo "=== Creating GPU VM: $VM_NAME ==="
    echo "    Size: $VM_SIZE (1x T4 GPU)"
    echo "    Location: $LOCATION"
    echo "    Image: $IMAGE"

    az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --size "$VM_SIZE" \
        --image "$IMAGE" \
        --admin-username "$ADMIN_USER" \
        --generate-ssh-keys \
        --public-ip-sku Standard \
        --os-disk-size-gb 128 \
        --output table

    echo ""
    echo "=== Opening SSH port ==="
    az vm open-port --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --port 22 -o none

    echo "Waiting 30s for VM to be ready..."
    sleep 30
fi

IP=$(get_ip)
if [[ -z "$IP" ]]; then
    echo "ERROR: Could not get VM IP. Is the VM running?"
    exit 1
fi
echo "VM IP: $IP"

# ── Upload code ──
echo "=== Uploading code to $VM_NAME ==="
cd "$REPO_ROOT"

tar czf /tmp/inftythink-code.tar.gz \
    --exclude='.venv' \
    --exclude='checkpoints' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='results/figures' \
    -C "$REPO_ROOT/.." "$(basename "$REPO_ROOT")"

scp -o StrictHostKeyChecking=no /tmp/inftythink-code.tar.gz "$ADMIN_USER@$IP":~
rm /tmp/inftythink-code.tar.gz

ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$IP" "rm -rf ~/reasoning && tar xzf ~/inftythink-code.tar.gz -C ~"

# ── Setup VM environment ──
echo "=== Setting up VM environment (CUDA + Python + deps) ==="
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$IP" "chmod +x ~/reasoning/scripts/azure_setup_vm.sh && bash ~/reasoning/scripts/azure_setup_vm.sh"

# ── Run training in tmux ──
echo "=== Starting training in tmux session 'train' ==="
ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$IP" \
    "which tmux || sudo apt-get install -y tmux && \
     chmod +x ~/reasoning/scripts/gcp_train.sh && \
     tmux new-session -d -s train 'bash ~/reasoning/scripts/gcp_train.sh 2>&1 | tee ~/reasoning/train.log'"

echo ""
echo "============================================"
echo " VM is training!"
echo " IP: $IP"
echo ""
echo " To monitor:"
echo "   bash scripts/azure_deploy.sh --ssh"
echo "   tmux attach -t train"
echo ""
echo " To download results when done:"
echo "   bash scripts/azure_deploy.sh --download"
echo ""
echo " To delete VM when done (IMPORTANT — stops billing):"
echo "   bash scripts/azure_deploy.sh --delete"
echo "============================================"
