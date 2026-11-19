#!/bin/bash
# One-shot setup for a fresh Lambda GPU instance (Ubuntu + Lambda Stack).
#   git clone https://github.com/evandiewald/weathernext-gpu && cd weathernext-gpu
#   bash scripts/setup_instance.sh
# Then reboot once (needed for the profiling-permission setting), and regenerate
# the reference data with:
#   .venv-jax/bin/python scripts/03_b0_jax_baseline.py
#   .venv-jax/bin/python -c "import numpy as np; np.save('data/parity_noise.npy', np.random.default_rng(0).standard_normal((1, 32)).astype(np.float32))"
#   .venv-jax/bin/python scripts/03_b0_jax_baseline.py --noise data/parity_noise.npy --matmul-precision highest --tag parity
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data logs results/profiles

# uv: fast installer that also provides Python 3.12 (system Python is 3.10; JAX needs >= 3.11)
python3 -m pip install --user -q uv
UV=~/.local/bin/uv

$UV venv -q --python 3.12 .venv-jax
$UV pip install -q --python .venv-jax/bin/python -r requirements-jax.txt
$UV venv -q --python 3.12 .venv-torch
$UV pip install -q --python .venv-torch/bin/python -r requirements-torch.txt

# Nsight Compute CLI (ncu) from Lambda's apt repo
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nsight-compute

# Let non-root users read GPU performance counters (Nsight GUI connects as ubuntu).
# Takes effect after a reboot.
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u

.venv-jax/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
.venv-torch/bin/python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo "done - now: sudo reboot"
