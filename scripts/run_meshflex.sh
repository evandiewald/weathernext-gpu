#!/bin/bash
# Full-mesh flex attention: parity (fp32, 1 timed step - flex without TF32 is slow)
# and timing (TF32, with and without torch.compile) for each node ordering.
cd "$(dirname "$0")/.."
PY=.venv-torch/bin/python
REF=data/b0_jax_WeatherNextCyclones_Mini_NVIDIA_A10_parity_prediction.nc
for order in "${@:-hilbert native}"; do
  for o in $order; do
    echo "=== $o parity (fp32)"
    $PY scripts/05_b1_torch.py --mesh-attn $o --noise data/parity_noise.npy --tag parity --n-timed 1 2>&1 | grep -E "^step|rror"
    $PY scripts/06_parity.py $REF data/b1_torch_mini_NVIDIA_A10_meshflex-${o}_fp32_parity_prediction.npz 2>&1 | tail -1
    echo "=== $o timing (tf32)"
    $PY scripts/05_b1_torch.py --mesh-attn $o --tf32 --noise data/parity_noise.npy --tag sweep 2>&1 | grep -E "^(step|peak)|rror"
    echo "=== $o timing (tf32 + compile)"
    $PY scripts/05_b1_torch.py --mesh-attn $o --tf32 --compile --noise data/parity_noise.npy --tag sweep 2>&1 | grep -E "^(step|peak)|rror"
  done
done
