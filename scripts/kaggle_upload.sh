#!/usr/bin/env bash
# Upload the InftyThink codebase to Kaggle as a dataset, then push the notebook as a kernel.
#
# Prerequisites:
#   1. pip install kaggle
#   2. Place your API token at ~/.kaggle/kaggle.json
#      (Download from kaggle.com -> Account -> API -> Create New Token)
#   3. Set KAGGLE_USERNAME to your Kaggle username, or it will be read from kaggle.json.
#
# Usage:
#   bash scripts/kaggle_upload.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_SLUG="inftythink-code"
NOTEBOOK_SLUG="inftythink-training"

# ---------------------------------------------------------------------------
# Resolve Kaggle username
# ---------------------------------------------------------------------------
if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    if command -v python3 &>/dev/null && [[ -f ~/.kaggle/kaggle.json ]]; then
        KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
    else
        echo "ERROR: Set KAGGLE_USERNAME or place ~/.kaggle/kaggle.json"
        exit 1
    fi
fi

echo "Kaggle user: ${KAGGLE_USERNAME}"
echo "Repo dir:    ${REPO_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Create / update the dataset with the code
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Uploading codebase as Kaggle dataset '${DATASET_SLUG}' ==="

STAGING_DIR=$(mktemp -d)
trap "rm -rf ${STAGING_DIR}" EXIT

# Copy code, excluding large/unnecessary dirs
rsync -a --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints/' \
    --exclude='results/' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='node_modules' \
    "${REPO_DIR}/" "${STAGING_DIR}/"

# Create dataset-metadata.json
cat > "${STAGING_DIR}/dataset-metadata.json" <<METAEOF
{
  "title": "InftyThink Code",
  "id": "${KAGGLE_USERNAME}/${DATASET_SLUG}",
  "licenses": [{"name": "MIT"}]
}
METAEOF

# Check if dataset already exists
if kaggle datasets list --mine --search "${DATASET_SLUG}" 2>/dev/null | grep -q "${DATASET_SLUG}"; then
    echo "Dataset exists, creating new version..."
    kaggle datasets version -p "${STAGING_DIR}" -m "Updated code" --dir-mode zip
else
    echo "Creating new dataset..."
    kaggle datasets create -p "${STAGING_DIR}" --dir-mode zip
fi

echo "Dataset upload complete."

# Wait for dataset to be processed
echo "Waiting for dataset processing (this may take a minute)..."
sleep 30

# ---------------------------------------------------------------------------
# Step 2: Push the notebook as a Kaggle kernel
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Pushing notebook as Kaggle kernel '${NOTEBOOK_SLUG}' ==="

KERNEL_DIR=$(mktemp -d)

# Copy the notebook
cp "${REPO_DIR}/notebooks/kaggle_inftythink.ipynb" "${KERNEL_DIR}/"

# Create kernel-metadata.json
cat > "${KERNEL_DIR}/kernel-metadata.json" <<METAEOF
{
  "id": "${KAGGLE_USERNAME}/${NOTEBOOK_SLUG}",
  "title": "InftyThink: Iterative Reasoning Training",
  "code_file": "kaggle_inftythink.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["${KAGGLE_USERNAME}/${DATASET_SLUG}"],
  "competition_sources": [],
  "kernel_sources": []
}
METAEOF

kaggle kernels push -p "${KERNEL_DIR}"

echo ""
echo "=== Done! ==="
echo ""
echo "Your notebook is now on Kaggle. Next steps:"
echo "  1. Go to: https://www.kaggle.com/${KAGGLE_USERNAME}/${NOTEBOOK_SLUG}"
echo "  2. Verify the dataset '${DATASET_SLUG}' is attached under 'Data'"
echo "  3. Confirm GPU accelerator and Internet are enabled in Settings"
echo "  4. Click 'Run All' to start the pipeline"
echo ""
echo "Alternatively, run the notebook interactively to monitor progress cell by cell."
