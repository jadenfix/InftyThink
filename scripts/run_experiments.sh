#!/usr/bin/env bash
# Run the full experiment suite in dependency order.
# Outputs: results/*.json, results/ablations/*.json, results/figures/*.png
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source .venv/bin/activate
# Only force CPU if no GPU available
if ! python -c "import jax; assert any('gpu' in str(d).lower() for d in jax.devices())" 2>/dev/null; then
    export JAX_PLATFORMS=cpu
fi

echo "=== [1/10] Baseline B1: Vanilla CoT ==="
python main.py run-experiment --name b1_vanilla_cot

echo "=== [2/10] Baseline B2: Capped CoT ==="
python main.py run-experiment --name b2_capped_cot

echo "=== [3/10] Baseline B3: Segmented, no summary ==="
python main.py run-experiment --name b3_segmented_no_summary

echo "=== [4/10] Baseline B4: Truncation ==="
python main.py run-experiment --name b4_truncation

echo "=== [5/10] Main Method M1: InftyThink ==="
python main.py run-experiment --name m1_inftythink

echo "=== [6/10] Ablation A1: Segment length ==="
python main.py run-experiment --name a1_segment_length

echo "=== [7/10] Ablation A2: Summary length ==="
python main.py run-experiment --name a2_summary_length

echo "=== [8/10] Ablation A3: Iterations ==="
python main.py run-experiment --name a3_iterations

echo "=== [9/10] Ablation A4: Conditioning ==="
python main.py run-experiment --name a4_conditioning

echo "=== [10/10] Extension E1: Structured state ==="
python main.py run-experiment --name e1_structured_state

echo "=== Generating figures ==="
python main.py analyze --results-dir results/

echo "=== All experiments complete. Results in results/ ==="
