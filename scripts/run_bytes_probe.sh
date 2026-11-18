#!/bin/bash
# Hardware byte counts for one whole forecast step, per variant ("attn compile|none").
# Output: results/profiles/bytes_<attn>[_compile].csv  -> feed to 07_bytes_per_step.py
set -u
cd "$(dirname "$0")/.."
NCU=/usr/lib/nsight-compute/ncu
for variant in "$@"; do
  read -r attn compile <<< "$variant"
  flag=""; suffix=""
  [ "$compile" = "compile" ] && flag="--compile" && suffix="_compile"
  out="results/profiles/bytes_${attn}${suffix}.csv"
  echo "=== $attn $compile -> $out"
  $NCU --nvtx --nvtx-include "step_probe/" --csv --print-units base \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    .venv-torch/bin/python scripts/05_b1_torch.py --tf32 --attn "$attn" $flag \
    --noise data/parity_noise.npy --nvtx-step --tag bytes > "$out" 2> "${out%.csv}.err"
  grep -c '"' "$out"
done
