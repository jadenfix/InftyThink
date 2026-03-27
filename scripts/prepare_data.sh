#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source .venv/bin/activate
export JAX_PLATFORMS=cpu

python main.py prepare-data \
    --n-train 5000 \
    --n-eval 500 \
    --seed 42 \
    --tokenizer gpt2
