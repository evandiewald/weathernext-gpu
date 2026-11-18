#!/bin/bash
# Phase 2 sweep: each attention variant twice.
#   fp32 + fixed noise  -> parity vs the JAX fp32 reference
#   tf32 + fixed noise  -> timing (TF32 is what B0/B1 timings use)
set -u
cd "$(dirname "$0")/.."
PY=.venv-torch/bin/python
REF=data/b0_jax_WeatherNextCyclones_Mini_NVIDIA_A10_parity_prediction.nc
for variant in "$@"; do
  read -r attn compile <<< "$variant"
  flag=""; [ "$compile" = "compile" ] && flag="--compile"
  echo "=== $attn $compile"
  $PY scripts/05_b1_torch.py --attn "$attn" $flag --noise data/parity_noise.npy --tag parity 2>&1 | grep -E "^step|Error|error" | head -5
  name="b1_torch_mini_NVIDIA_A10_${attn}$([ -n "$flag" ] && echo _compile)_fp32_parity"
  $PY scripts/06_parity.py "$REF" "data/${name}_prediction.npz" 2>&1 | tail -1
  $PY scripts/05_b1_torch.py --attn "$attn" $flag --tf32 --noise data/parity_noise.npy --tag sweep 2>&1 | grep -E "^(step|peak)|Error|error" | head -5
done
