"""Step 11: FlexAttention over the whole mesh in different node orderings.

Instead of the port's layout (node blocks of size = RCM bandwidth, each attending
to itself + 2 neighbours), run ONE flex call over all N mesh nodes with the exact
k-hop mask, in a chosen node ordering. Same math per node (attention only depends
on WHICH nodes are neighbours, not their order), different tile structure.

For each ordering: A = mask_mod looks up the dense N x N mask (exact),
                   B = same tiles, no lookup (wrong on purpose: the floor).
Needs results/order_<name>_splits5.npy from scripts/10_ordering_analysis.py.

    python scripts/11_ordering_flex_bench.py --dtype fp32|bf16 [--block 128]
"""
import argparse
import importlib.util
import pathlib

import numpy as np
import scipy.sparse as sps
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("orders", here / "10_ordering_analysis.py")
orders_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orders_mod)

flex = torch.compile(flex_attention)


def bench(fn, n=20) -> float:
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
    im = orders_mod.icosahedral_mesh
    mesh = im.get_last_triangular_mesh_for_sphere(5)
    n = len(mesh.vertices)
    adj = im.get_sparse_adjacency_matrix(mesh).astype(np.float32)
    adj = (adj + sps.eye(n, dtype=np.float32)).tocsr()
    pairs = list(orders_mod.khop_rows(adj, 16))
    rows = torch.from_numpy(np.concatenate([p[0] for p in pairs])).to(dev)
    cols = torch.from_numpy(np.concatenate([p[1] for p in pairs])).to(dev)
    useful = rows.numel()

    H, D = 4, 128
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, H, n, D, device=dev, dtype=dtype, generator=g)
    k = torch.randn(1, H, n, D, device=dev, dtype=dtype, generator=g)
    v = torch.randn(1, H, n, D, device=dev, dtype=dtype, generator=g)
    flops = 4 * useful * H * D
    print(f"{a.dtype}, one layer, {n} nodes x {H} heads, block {a.block}; port layout for reference: "
          f"17.4 ms fp32 / 10.2 ms bf16 (A), 4.8 / 1.2 ms (B)")
    for name in ["rcm", "hilbert"]:
        perm = torch.from_numpy(np.load(f"results/order_{name}_splits5.npy").astype(np.int64)).to(dev)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n, device=dev)
        dense = torch.zeros(n, n, dtype=torch.bool, device=dev)
        dense[inv[rows], inv[cols]] = True          # mask in this ordering

        def mod(b, h, q_idx, kv_idx):
            return dense[q_idx, kv_idx]
        bm = create_block_mask(mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=dev,
                               BLOCK_SIZE=a.block)
        bm_free = BlockMask.from_kv_blocks(bm.kv_num_blocks, bm.kv_indices, BLOCK_SIZE=bm.BLOCK_SIZE,
                                           mask_mod=None, seq_lengths=(n, n))
        tiles = int(bm.kv_num_blocks.sum()) + int(bm.full_kv_num_blocks.sum() if bm.full_kv_num_blocks is not None else 0)
        with torch.inference_mode():
            t_a = bench(lambda: flex(q, k, v, block_mask=bm))
            t_b = bench(lambda: flex(q, k, v, block_mask=bm_free))
        print(f"  {name:8s} tiles {tiles:5d}   A exact {t_a:6.2f} ms ({flops / t_a / 1e9:4.1f} TFLOP/s)   "
              f"B no-lookup {t_b:6.2f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--block", type=int, default=128)
    main(ap.parse_args())
