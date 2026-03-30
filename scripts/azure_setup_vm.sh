#!/usr/bin/env bash
# Run ON the Azure VM: install CUDA drivers, Python, deps.
set -euo pipefail

echo "=== Installing NVIDIA drivers + CUDA ==="
if ! command -v nvidia-smi &>/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y linux-headers-$(uname -r) build-essential

    # Install NVIDIA driver + CUDA toolkit via Ubuntu's nvidia-driver-meta
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-4 cuda-drivers

    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi

echo "=== GPU check ==="
nvidia-smi || echo "WARNING: nvidia-smi not available yet (may need reboot)"

echo "=== Installing uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

cd ~/reasoning

echo "=== Creating venv ==="
uv venv .venv
source .venv/bin/activate

echo "=== Installing GPU deps ==="
uv pip install -r requirements-gpu.txt

echo "=== JAX device check ==="
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print('GPU OK')
else:
    print('WARNING: No GPU detected by JAX — falling back to CPU')
    print('You may need to reboot the VM for NVIDIA drivers to load.')
"

echo "=== VM setup complete ==="
