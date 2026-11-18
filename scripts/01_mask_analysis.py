"""Step 1: how much attention work is "wasted" by each implementation?

CPU-only, no GPU or weights needed. Uses the official mesh code directly.

Setup (once):
    git clone --depth 1 --branch v0.3.0 \
        https://github.com/google-deepmind/weathernext.git third_party/weathernext
    pip install numpy scipy

Run:
    python scripts/01_mask_analysis.py --splits 5 --khop 16   # Mini   (~5 s)
    python scripts/01_mask_analysis.py --splits 6 --khop 32   # full   (~1-2 min)

Background:
  * The processor is a transformer over mesh nodes. Node i may only attend to
    node j if j is within `khop` hops of i on the mesh. That's the attention
    MASK: an N x N true/false matrix, mostly false.
  * "Useful work" = number of True entries (score computations we actually need).
  * Every implementation computes MORE than that, because GPUs/TPUs work on
    rectangular tiles, not individual entries. The ratio
    (work actually done) / (useful work) is our "waste factor".
"""
import argparse
import importlib.util
import pathlib
import time
import warnings

import numpy as np
import scipy.sparse as sps

warnings.filterwarnings("ignore")  # scipy sparse-efficiency chatter

# Import just the mesh module from the reference repo. It only needs numpy and
# scipy, so we skip installing JAX on the Mac.
REPO = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "weathernext"
spec = importlib.util.spec_from_file_location(
    "icosahedral_mesh", REPO / "weathernext" / "utils" / "icosahedral_mesh.py")
icosahedral_mesh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icosahedral_mesh)


def main(splits: int, khop: int, tile: int) -> None:
    # 1. Build the mesh exactly like the model does, including its node
    #    reordering (reverse Cuthill-McKee, "RCM": renumbers nodes so that
    #    neighbours get nearby indices -> mask entries hug the diagonal).
    mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits)
    perm, _ = icosahedral_mesh.get_permutation_to_banded(mesh)
    adj = icosahedral_mesh.get_sparse_adjacency_matrix(mesh).astype(np.float32)
    n = adj.shape[0]
    adj = (adj + sps.eye(n, dtype=np.float32))[perm][:, perm].tocsr()

    # 2. Build the k-hop mask one chunk of rows at a time.
    #    Multiplying a one-hot row by (A + I) k times = "everything reachable
    #    in <= k hops". Doing all N rows at once needs too much RAM.
    useful = 0
    bandwidth = 0  # max |i - j| over True entries
    n_tiles = -(-n // tile)
    tile_counts = np.zeros((n_tiles, n_tiles), dtype=np.int32)  # needed cells per tile
    t0 = time.time()
    for start in range(0, n, 1024):
        rows = np.arange(start, min(n, start + 1024))
        x = sps.csr_matrix((np.ones(len(rows), np.float32),
                            (np.arange(len(rows)), rows)), shape=(len(rows), n))
        for _ in range(khop):
            x = x @ adj
            x.data[:] = 1  # keep it boolean
        x = x.tocoo()
        r, c = rows[x.row], x.col
        useful += len(r)
        bandwidth = max(bandwidth, int(np.abs(r - c).max()) + 1)
        np.add.at(tile_counts, (r // tile, c // tile), 1)

    # 3. Cost of each implementation, in "score entries computed".
    dense = n * n
    #    GPU path ('triblockdiag_mha'): cut nodes into blocks of size
    #    `bandwidth`; each block attends to itself + the block on each side.
    n_blocks = -(-n // bandwidth)
    triblockdiag = n_blocks * 3 * bandwidth ** 2
    #    TPU path ('splash_mha'): compute only tiles with >= 1 True entry.
    splash = int((tile_counts > 0).sum()) * tile * tile

    print(f"mesh: {n} nodes, {khop}-hop window")
    print(f"  avg neighbours per node : {useful / n:,.0f}  "
          f"(mask density {useful / dense:.2%})")
    print(f"  RCM bandwidth           : {bandwidth:,}")
    print(f"  waste factor  dense       : {dense / useful:5.1f}x")
    print(f"  waste factor  triblockdiag: {triblockdiag / useful:5.1f}x   <- GPU default")
    print(f"  waste factor  splash {tile}  : {splash / useful:5.1f}x   <- TPU default")
    print(f"  ({time.time() - t0:.0f}s)")

    out = pathlib.Path("results") / f"tiles_splits{splits}_k{khop}_t{tile}.npz"
    np.savez(out, tile_counts=tile_counts, n=n, khop=khop, tile=tile,
             bandwidth=bandwidth, useful=useful)
    print(f"saved tile counts -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--khop", type=int, default=16)
    p.add_argument("--tile", type=int, default=128)
    a = p.parse_args()
    main(a.splits, a.khop, a.tile)
