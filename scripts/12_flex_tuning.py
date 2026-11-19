"""Step 12: tune FlexAttention's kernel settings for the Hilbert-ordered mesh mask.

The profile showed the kernel is latency-bound: 1 block (4 warps) per SM because
128x128 fp32 tiles fill shared memory, 8% occupancy. Knobs (all exact, no math change):
  mask block   : tile size of the block mask (64 skips more empty space than 128)
  BLOCK_M/N    : query / key tile the kernel works on (must divide the mask block)
  num_stages   : how many key tiles are prefetched (more = more shared memory)
  num_warps    : warps per thread block

    python scripts/12_flex_tuning.py --dtype fp32    # TF32 on
Writes results/flex_tuning_<dtype>.csv
"""
import argparse
import csv
import importlib.util
import itertools
import pathlib

import numpy as np
import scipy.sparse as sps
import torch
import torch._dynamo
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch._dynamo.config.cache_size_limit = 256      # one compiled kernel per config
torch._dynamo.config.recompile_limit = 256

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("orders", here / "10_ordering_analysis.py")
orders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orders)
flex = torch.compile(flex_attention)


def bench(fn, n=15) -> float:
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    im = orders.icosahedral_mesh
    mesh = im.get_last_triangular_mesh_for_sphere(5)
    n = len(mesh.vertices)
    adj = (im.get_sparse_adjacency_matrix(mesh).astype(np.float32) + sps.eye(n, dtype=np.float32)).tocsr()
    pairs = list(orders.khop_rows(adj, 16))
    rows = torch.from_numpy(np.concatenate([p[0] for p in pairs])).to(dev)
    cols = torch.from_numpy(np.concatenate([p[1] for p in pairs])).to(dev)
    perm = torch.from_numpy(np.argsort(orders.hilbert_key(mesh.vertices), kind="stable").astype(np.int64)).to(dev)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=dev)
    dense = torch.zeros(n, n, dtype=torch.bool, device=dev)
    dense[inv[rows], inv[cols]] = True

    def mod(b, h, q_idx, kv_idx):
        return dense[q_idx, kv_idx]

    H, D = 4, 128
    g = torch.Generator(device=dev).manual_seed(0)
    q, k, v = (torch.randn(1, H, n, D, device=dev, dtype=dtype, generator=g) for _ in range(3))
    masks = {mb: create_block_mask(mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=dev, BLOCK_SIZE=mb)
             for mb in (128, 64)}
    with torch.inference_mode():
        ref = flex(q, k, v, block_mask=masks[128]).float()

    configs = []
    for mb, bm_, bn, st, w in itertools.product((128, 64), (128, 64, 32), (128, 64, 32), (1, 2, 3), (4, 8)):
        if bm_ <= mb and bn <= mb:
            configs.append((mb, bm_, bn, st, w))
    out = pathlib.Path(f"results/flex_tuning_{a.dtype}.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mask_block", "BLOCK_M", "BLOCK_N", "num_stages", "num_warps", "ms", "max_abs_diff"])
        for mb, bm_, bn, st, w in configs:
            opts = {"BLOCK_M": bm_, "BLOCK_N": bn, "num_stages": st, "num_warps": w}
            try:
                with torch.inference_mode():
                    fn = lambda: flex(q, k, v, block_mask=masks[mb], kernel_options=opts)
                    diff = (fn().float() - ref).abs().max().item()
                    ms = bench(fn)
            except Exception as ex:                       # e.g. out of shared memory
                ms, diff = float("nan"), str(type(ex).__name__)
            wr.writerow([mb, bm_, bn, st, w, f"{ms:.3f}", diff if isinstance(diff, str) else f"{diff:.2e}"])
            f.flush()
            print(f"mask {mb:3d}  M {bm_:3d} N {bn:3d} stages {st} warps {w}:  {ms:7.2f} ms  diff {diff}", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    main(ap.parse_args())
