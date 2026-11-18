"""Step 7: how many bytes does one forecast step move, and where?

Parses ncu CSV output captured around one step (NVTX range "step_probe"):
    ncu --nvtx --nvtx-include "step_probe/" --csv --print-units base \\
        --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \\
        python scripts/05_b1_torch.py --tf32 [--attn X] [--compile] --nvtx-step > out.csv

    python scripts/07_bytes_per_step.py out.csv [more.csv ...]

Note: ncu serializes kernels and locks clocks, so its times are a bit higher than
the wall-clock step times; the bytes are exact hardware counts.
"""
import collections
import csv
import io
import re
import sys


def kernel_kind(name: str) -> str:
    """Coarse family of a kernel, from its (demangled) name."""
    n = name.lower()
    if "softmax" in n:
        return "softmax"
    if "flex_attention" in n or "flash" in n or "fmha" in n or "attention" in n:
        return "fused attention"
    if "gemm" in n or "kernel2" in n or "cutlass" in n or "sm80_xmma" in n or "ampere" in n:
        return "matmul (cuBLAS)"
    if n.startswith("triton_"):
        return "triton (torch.compile)"
    if "elementwise" in n or "copy" in n:
        return "elementwise / copy"
    if "reduce" in n or "norm" in n:
        return "reduction / norm"
    return "other"


def load(path: str):
    lines = [l for l in open(path) if l.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("".join(lines))))
    kernels = collections.OrderedDict()
    for r in rows:
        k = kernels.setdefault(r["ID"], {"name": r["Kernel Name"]})
        k[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
    return kernels


def main(paths: list[str]) -> None:
    for path in paths:
        kernels = load(path)
        by_kind = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for k in kernels.values():
            kind = kernel_kind(k["name"])
            t = k["gpu__time_duration.sum"] / 1e6
            b = (k["dram__bytes_read.sum"] + k["dram__bytes_write.sum"]) / 1e9
            by_kind[kind][0] += t
            by_kind[kind][1] += b
            by_kind[kind][2] += 1
        tt = sum(v[0] for v in by_kind.values())
        tb = sum(v[1] for v in by_kind.values())
        print(f"\n{path}\n  {len(kernels)} kernels, {tt:.1f} ms, {tb:.1f} GB DRAM traffic "
              f"(avg {tb / tt * 1e3:.0f} GB/s)")
        for kind, (t, b, n) in sorted(by_kind.items(), key=lambda x: -x[1][1]):
            print(f"  {kind:26s} {n:5d} kernels {t:8.1f} ms {b:8.1f} GB")


if __name__ == "__main__":
    main(sys.argv[1:])
