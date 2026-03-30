#!/usr/bin/env bash
# Run ON the VM: activate env, run full pipeline (data prep + train + experiments + analysis).
set -euo pipefail

cd ~/reasoning
source .venv/bin/activate
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

echo "=== [1/4] Preparing data ==="
python main.py prepare-data --n-train 5000 --n-eval 500

echo "=== [2/4] Training model ==="
python main.py train --config configs/base.yaml

echo "=== [3/4] Running experiments ==="
bash scripts/run_experiments.sh

echo "=== [4/4] Generating analysis ==="
python main.py analyze --results-dir results/

echo "=== ALL DONE ==="
echo "Results in ~/reasoning/results/"
