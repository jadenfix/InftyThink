#!/usr/bin/env bash
# Run ON the VM after SSH: installs CUDA drivers, Python, deps, and uploads code.
set -euo pipefail

echo "=== Installing NVIDIA drivers ==="
if ! command -v nvidia-smi &>/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y linux-headers-$(uname -r)
    # Install CUDA 12.4 toolkit + drivers
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-4 cuda-drivers
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi

echo "=== GPU check ==="
nvidia-smi

echo "=== Installing uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

cd ~/reasoning

echo "=== Creating venv ==="
uv venv .venv
source .venv/bin/activate

echo "=== Installing GPU deps ==="
uv pip install -r requirements-gpu.txt

echo "=== JAX GPU smoke test ==="
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices), \
    f'No GPU found! Devices: {devices}'
print('GPU OK')
"

echo "=== VM setup complete ==="
