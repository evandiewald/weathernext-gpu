"""Step 4: turn a JAX profiler trace into "where does a forecast step's time go".

The trace lists every GPU kernel with its duration and the XLA instruction it
came from (`hlo_op`). The optimized-HLO dump maps each instruction back to the
model code that produced it (`op_name`, e.g. .../mesh_transformer/.../ffw_up).
Join the two, bucket by model stage, and sum.

Capture both on the GPU box (command buffers OFF, otherwise kernels are hidden
inside CUDA graphs and lose their `hlo_op`):
    XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_dump_to=/tmp/hlo \
      --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=^$" \
      python scripts/03_b0_jax_baseline.py --profile results/profiles/b0

Then:
    python scripts/04_profile_breakdown.py <trace.json.gz> <*jit__lambda*after_optimizations.txt>
"""
import collections
import csv
import gzip
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt

STEPS = 3  # forecast steps recorded inside the profiler window

# (stage, rule) checked in order; rule is a substring of the op_name path.
ATTN_SCORES = ("bnqhd,bnkhd->bnhqk",)
ATTN_AV = ("bnhqk,bnkhd->bnqhd",)
ATTN_SOFTMAX = ("jit(_where)", "reduce_max", "reduce_sum", "/exp", "/div", "/sub")
ATTN_PROJ = ("mha_proj_", "mha_final")
FFW = ("ffw_up", "ffw_down", "tanh", "integer_pow")
NORM = ("norm",)


def stage(op_name: str) -> str:
    if "grid_to_mesh_gnn" in op_name or "grid_encoder" in op_name \
            or "mesh_encoder" in op_name or "norm_conditioning_encoder" in op_name:
        return "encoder (grid -> mesh)"
    if "mesh_to_grid_gnn" in op_name or "grid_decoder" in op_name:
        return "decoder (mesh -> grid)"
    if "mesh_transformer" in op_name:
        if any(r in op_name for r in ATTN_SCORES):
            return "attention: scores QK^T"
        if any(r in op_name for r in ATTN_AV):
            return "attention: weights x V"
        if any(r in op_name for r in ATTN_SOFTMAX):
            return "attention: mask + softmax"
        if any(r in op_name for r in ATTN_PROJ):
            return "attention: Q/K/V/out projections"
        if any(r in op_name for r in FFW):
            return "feed-forward"
        if any(r in op_name for r in NORM):
            return "norm + noise conditioning"
        return "transformer: other"
    return "other"


def main(trace_path: str, hlo_path: str) -> None:
    hlo = pathlib.Path(hlo_path).read_text()
    op_names = dict(re.findall(
        r'^\s*%?([\w.\-]+) = [^\n]*?metadata=\{op_name="([^"]*)"', hlo, re.M))

    trace = json.load(gzip.open(trace_path))
    procs = {e["pid"]: e["args"]["name"] for e in trace["traceEvents"]
             if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu = {p for p, n in procs.items() if "GPU" in n}
    kernels = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e["pid"] in gpu]

    by_stage = collections.Counter()
    by_kernel = collections.Counter()
    kernel_stage = {}
    for e in kernels:
        op = e["args"].get("hlo_op", "")
        s = "memcpy" if "Memcpy" in e["name"] else stage(op_names.get(op, ""))
        by_stage[s] += e["dur"]
        by_kernel[op or e["name"]] += e["dur"]
        kernel_stage[op or e["name"]] = s

    total = sum(by_stage.values())
    print(f"GPU busy time: {total / STEPS / 1e3:.1f} ms per step ({len(kernels) // STEPS} kernels/step)\n")
    print(f"{'stage':36s} {'ms/step':>8s} {'share':>6s}")
    rows = []
    for s, d in by_stage.most_common():
        print(f"{s:36s} {d / STEPS / 1e3:8.1f} {d / total:6.1%}")
        rows.append((s, round(d / STEPS / 1e3, 2), round(d / total, 4)))
    # The same kernel runs once per layer (loop_divide_fusion.1, .2, ...):
    # strip the numeric suffix to group the copies, count launches per step.
    grouped, launches = collections.Counter(), collections.Counter()
    for e in kernels:
        op = e["args"].get("hlo_op", "") or e["name"]
        key = (re.sub(r"\.\d+$", "", op), kernel_stage[op])
        grouped[key] += e["dur"]
        launches[key] += 1
    print("\ntop kernels (summed over layers):")
    for k, d in grouped.most_common(10):
        print(f"  {d / STEPS / 1e3:7.1f} ms  x{launches[k] // STEPS:<3d} {k[0][:26]:26s} {k[1]}")

    out = pathlib.Path(trace_path).parents[3] / "breakdown"
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        csv.writer(f).writerows([("stage", "ms_per_step", "share"), *rows])

    # Bar chart: one hue, sorted, direct labels.
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#2a78d6", height=0.6)
    for i, r in enumerate(rows):
        ax.text(r[1] + total / STEPS / 1e3 * 0.01, i, f"{r[1]:.0f} ms ({r[2]:.0%})",
                va="center", fontsize=10, color="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("GPU time per forecast step (ms)", color="#52514e")
    ax.set_title(f"Where a forecast step goes: {total / STEPS / 1e3:.0f} ms of GPU time",
                 loc="left", fontsize=12)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor="#fcfcfb")
    print(f"\nsaved {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
