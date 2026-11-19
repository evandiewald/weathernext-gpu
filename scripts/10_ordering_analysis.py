"""Step 10: does a different node numbering make the attention mask tile-friendlier?

CPU only. For the Mini mesh (10,242 nodes, 16-hop window) and each candidate
ordering, tile the FULL N x N mask into 128 x 128 tiles and count:
  computed tiles : tiles with at least one needed entry (a block-sparse kernel runs these)
  full tiles     : tiles where EVERY entry is needed -> no per-element mask lookup at all
  waste          : computed entries / needed entries

Orderings:
  rcm      reverse Cuthill-McKee (what the model uses; minimizes bandwidth)
  latlon   sort by latitude band, then longitude (naive "row by row")
  morton   Z-order curve over the nodes' 3D positions on the sphere
  hilbert  Hilbert curve over the same 3D positions (no big jumps, unlike Z)

    python scripts/10_ordering_analysis.py [--splits 5 --khop 16 --tile 128]
"""
import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import scipy.sparse as sps

REPO = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "weathernext"
spec = importlib.util.spec_from_file_location(
    "icosahedral_mesh", REPO / "weathernext" / "utils" / "icosahedral_mesh.py")
icosahedral_mesh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icosahedral_mesh)

BITS = 10  # grid resolution per axis for the space-filling curves (1024^3 cells)


def quantize(xyz: np.ndarray) -> np.ndarray:
    """Map points on the unit sphere to integer grid coords in [0, 2^BITS)."""
    return np.clip(((xyz + 1) / 2 * (2**BITS - 1)).round(), 0, 2**BITS - 1).astype(np.int64)


def morton_key(xyz: np.ndarray) -> np.ndarray:
    """Interleave the bits of x, y, z: nearby keys -> nearby points (mostly)."""
    c = quantize(xyz)
    key = np.zeros(len(c), dtype=np.int64)
    for b in range(BITS):
        for axis in range(3):
            key |= ((c[:, axis] >> b) & 1) << (3 * b + axis)
    return key


def hilbert_key(xyz: np.ndarray) -> np.ndarray:
    """3D Hilbert index (Skilling's 'transpose' algorithm). Like Morton, but the
    curve never jumps: consecutive keys are always adjacent grid cells."""
    out = np.zeros(len(xyz), dtype=np.int64)
    for i, p in enumerate(quantize(xyz)):
        x = [int(v) for v in p]
        m = 1 << (BITS - 1)
        q = m
        while q > 1:                       # inverse undo excess work
            pmask = q - 1
            for j in range(3):
                if x[j] & q:
                    x[0] ^= pmask
                else:
                    t = (x[0] ^ x[j]) & pmask
                    x[0] ^= t
                    x[j] ^= t
            q >>= 1
        for j in range(1, 3):              # Gray encode
            x[j] ^= x[j - 1]
        t = 0
        q = m
        while q > 1:
            if x[2] & q:
                t ^= q - 1
            q >>= 1
        for j in range(3):
            x[j] ^= t
        key = 0                            # interleave the transposed bits
        for b in range(BITS - 1, -1, -1):
            for j in range(3):
                key = (key << 1) | ((x[j] >> b) & 1)
        out[i] = key
    return out


def khop_rows(adj: sps.csr_matrix, k: int):
    """Yield (row indices, column indices) of the k-hop mask, 1024 rows at a time."""
    n = adj.shape[0]
    for start in range(0, n, 1024):
        rows = np.arange(start, min(n, start + 1024))
        x = sps.csr_matrix((np.ones(len(rows), np.float32), (np.arange(len(rows)), rows)), shape=(len(rows), n))
        for _ in range(k):
            x = x @ adj
            x.data[:] = 1
        x = x.tocoo()
        yield rows[x.row], x.col


def main(splits: int, khop: int, tile: int) -> None:
    mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits)
    xyz = mesh.vertices
    n = len(xyz)
    adj = icosahedral_mesh.get_sparse_adjacency_matrix(mesh).astype(np.float32)
    adj = (adj + sps.eye(n, dtype=np.float32)).tocsr()
    t0 = time.time()
    pairs = [(r, c) for r, c in khop_rows(adj, khop)]
    rows = np.concatenate([p[0] for p in pairs])
    cols = np.concatenate([p[1] for p in pairs])
    useful = len(rows)
    print(f"mesh {n} nodes, {khop}-hop: {useful / n:.0f} neighbours/node, density {useful / n / n:.2%} "
          f"({time.time() - t0:.0f}s)")

    lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    orders = {
        "rcm": icosahedral_mesh.get_permutation_to_banded(mesh)[0],
        "latlon": np.lexsort((lon, np.round(np.degrees(lat)))),
        "morton": np.argsort(morton_key(xyz), kind="stable"),
        "hilbert": np.argsort(hilbert_key(xyz), kind="stable"),
    }
    nt = -(-n // tile)
    sizes = np.minimum(tile, n - np.arange(nt) * tile)
    results = {}
    print(f"\n{tile}x{tile} tiles, {nt}x{nt} = {nt * nt} total")
    print(f"{'order':8s} {'computed':>9s} {'full':>6s} {'partial':>8s} {'waste':>6s}  {'mask lookups':>12s}")
    for name, perm in orders.items():
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = np.arange(n)           # node id -> position in this ordering
        counts = np.zeros((nt, nt), dtype=np.int64)
        np.add.at(counts, (inv[rows] // tile, inv[cols] // tile), 1)
        computed = counts > 0
        full = counts == np.outer(sizes, sizes)
        partial = computed & ~full
        waste = computed.sum() * tile * tile / useful
        lookups = partial.sum() * tile * tile / useful   # per needed entry
        print(f"{name:8s} {computed.sum():9d} {full.sum():6d} {partial.sum():8d} {waste:5.2f}x  {lookups:11.2f}x")
        results[name] = dict(computed=int(computed.sum()), full=int(full.sum()),
                             partial=int(partial.sum()), waste=waste, lookups_per_useful=lookups)
        np.save(pathlib.Path("results") / f"order_{name}_splits{splits}.npy", perm)
    out = pathlib.Path("results") / f"ordering_splits{splits}_k{khop}_t{tile}.json"
    out.write_text(json.dumps({"n": n, "khop": khop, "tile": tile, "useful": useful, "orders": results}, indent=2))
    print(f"\nsaved {out} and results/order_<name>_splits{splits}.npy (permutations)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--khop", type=int, default=16)
    ap.add_argument("--tile", type=int, default=128)
    a = ap.parse_args()
    main(a.splits, a.khop, a.tile)
