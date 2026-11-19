#!/bin/bash
# Full 0.25-degree model on the A10 (24 GB): does it fit, before vs after?
# Run after setup_instance.sh + the Mini reference runs (data/parity_noise.npy, Mini JAX fp32 ref).
#   bash scripts/run_full_model.sh            # steps 0-4 (~30-40 min incl. downloads)
#   bash scripts/run_full_model.sh all        # + compile timing and the fp32 parity check (slow)
# Every step writes results/<tag>.json, including the OOM ones (status "oom").
# Snapshots (results/profiles/*.pickle): drag into https://pytorch.org/memory_viz
cd "$(dirname "$0")/.."
TORCH=.venv-torch/bin/python
JAX=.venv-jax/bin/python
NOISE=data/parity_noise.npy
FLEX="--mesh-attn hilbert --flex-tuned --tf32 --noise $NOISE"
mkdir -p results/profiles

echo "=== 0. chunking unit test (CPU, seconds)"
$TORCH scripts/test_chunked_graph.py | tail -2 || exit 1

echo "=== 1. Mini with forced small chunks: same answer, cost of chunking"
#    expect: fp32 parity ~1.7e-5 (unchanged), TF32 step within a few ms of 132
$TORCH scripts/05_b1_torch.py --mesh-attn hilbert --flex-tuned --chunk 8192 --noise $NOISE \
    --tag parity --n-timed 1 2>&1 | grep -E "^(step|peak)|rror"
$TORCH scripts/06_parity.py data/b0_jax_WeatherNextCyclones_Mini_NVIDIA_A10_parity_prediction.nc \
    data/b1_torch_mini_NVIDIA_A10_meshflex-hilbert-tuned_fp32_chunk8192_parity_prediction.npz | tail -1
$TORCH scripts/05_b1_torch.py $FLEX --compile --chunk 8192 --tag sweep 2>&1 | grep -E "^(step|peak)|rror"

echo "=== 2. B0 (official JAX GPU path) on the full model"
#    expect: OOM; the saved error shows the size XLA tried to allocate
$JAX scripts/03_b0_jax_baseline.py --model full 2>&1 | grep -E "OUT OF MEMORY|^step|^peak|saved"

echo "=== 3. Optimized attention, decoder NOT chunked"
#    expect: OOM in mesh_to_grid (snapshot shows ~9.6 GB edge-message blocks)
$TORCH scripts/05_b1_torch.py --model full $FLEX --n-timed 5 \
    --memory-snapshot results/profiles/full_unchunked_snapshot.pickle 2>&1 | grep -E "^(step|peak)|OUT OF|rror|snapshot"

echo "=== 4. Optimized attention + chunked grid layers"
#    expect: fits; steady-state peak ~10-14 GB
$TORCH scripts/05_b1_torch.py --model full $FLEX --chunk 65536 --n-timed 5 \
    --memory-snapshot results/profiles/full_chunked_snapshot.pickle 2>&1 | grep -E "^(step|peak)|OUT OF|rror|snapshot"

[ "$1" = "all" ] || exit 0

echo "=== 5. + torch.compile (compile takes several minutes at this size)"
$TORCH scripts/05_b1_torch.py --model full $FLEX --chunk 65536 --compile --n-timed 5 2>&1 | grep -E "^(step|peak)|OUT OF|rror"

echo "=== 6. fp32 parity for the full model"
#    No GPU reference exists (B0 OOMs), so the reference is official JAX on the CPU in full fp32.
#    Needs ~100+ GB host RAM (gpu_1x_a10 has ~200 GB) and some minutes per step; torch side is
#    strict-fp32 flex (no tensor cores), also minutes. Expect ~1e-5 like Mini.
JAX_PLATFORMS=cpu $JAX scripts/03_b0_jax_baseline.py --model full --matmul-precision highest \
    --noise $NOISE --tag parity --n-timed 1 2>&1 | grep -E "^step|saved|OUT OF|rror"
$TORCH scripts/05_b1_torch.py --model full --mesh-attn hilbert --flex-tuned --chunk 65536 --noise $NOISE \
    --tag parity --n-timed 1 2>&1 | grep -E "^(step|peak)|OUT OF|rror"
$TORCH scripts/06_parity.py data/b0_jax_WeatherNextCyclones_cpu_parity_prediction.nc \
    data/b1_torch_full_NVIDIA_A10_meshflex-hilbert-tuned_fp32_chunk65536_parity_prediction.npz | tail -15
