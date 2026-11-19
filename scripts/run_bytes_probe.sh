#!/bin/bash
# Hardware byte counts for one whole forecast step, per variant.
# Each argument is "label=flags for 05_b1_torch.py", e.g.
#   scripts/run_bytes_probe.sh "eager=" "eager_compile=--compile" \
#                              "hilbert_compile=--mesh-attn hilbert --compile"
# Output: results/profiles/bytes_<label>.csv  -> feed to 07_bytes_per_step.py
set -u
cd "$(dirname "$0")/.."
NCU=/usr/lib/nsight-compute/ncu
for item in "$@"; do
  label=${item%%=*}; flags=${item#*=}
  out="results/profiles/bytes_${label}.csv"
  echo "=== $label ($flags) -> $out"
  $NCU --nvtx --nvtx-include "step_probe/" --csv --print-units base \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    .venv-torch/bin/python scripts/05_b1_torch.py --tf32 $flags \
    --noise data/parity_noise.npy --nvtx-step --tag bytes > "$out" 2> "${out%.csv}.err"
  grep -c '"' "$out"
done
