#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin on Linux/macOS (cargo path is legacy)
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Create venv
uv venv .venv
echo "Created .venv"

# Activate
source .venv/bin/activate

# Install deps
uv pip install -r requirements.txt
echo "Installed all dependencies"

# Set JAX to CPU
export JAX_PLATFORMS=cpu

# Smoke test
python -c "
import os; os.environ['JAX_PLATFORMS']='cpu'
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
assert str(devices[0]).startswith('TFRT_CPU') or 'cpu' in str(devices[0]).lower(), f'Expected CPU device, got {devices}'
print('Environment OK')
"
