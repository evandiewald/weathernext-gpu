"""Step 6: does the PyTorch port produce the same forecast as the official JAX code?

Compares every output variable of one 6-hour step, in physical units.
    python scripts/06_parity.py data/<jax>_prediction.nc data/<torch>_prediction.npz

Metric per variable: RMS(torch - jax) / std(jax field). That's the error as a
fraction of the field's natural variability, so it's comparable across
variables with different units (K, m/s, Pa, ...). Max abs diff in native units too.
"""
import sys

import numpy as np
import xarray


def main(jax_path: str, torch_path: str) -> None:
    ref = xarray.load_dataset(jax_path)
    test = np.load(torch_path)
    rows, missing = [], []
    for var in sorted(test.files):
        if var not in ref:
            missing.append(var)
            continue
        j = ref[var].squeeze(drop=True).values.astype(np.float64)
        t = np.squeeze(test[var]).astype(np.float64)
        if j.shape != t.shape:
            print(f"shape mismatch {var}: jax {j.shape} torch {t.shape}")
            continue
        ok = np.isfinite(j) & np.isfinite(t)
        nan_mismatch = int((np.isfinite(j) != np.isfinite(t)).sum())
        d = (t - j)[ok]
        scale = np.std(j[ok]) or 1.0
        rows.append((var, np.sqrt(np.mean(d ** 2)) / scale, np.abs(d).max(), nan_mismatch))

    rows.sort(key=lambda r: -r[1])
    print(f"{'variable':36s} {'rel RMS err':>12s} {'max |diff|':>12s} {'NaN mismatch':>12s}")
    for var, rel, mx, nm in rows:
        print(f"{var:36s} {rel:12.2e} {mx:12.3e} {nm:12d}")
    worst = rows[0][1] if rows else float("nan")
    print(f"\n{len(rows)} variables compared; worst relative RMS error {worst:.2e}")
    if missing:
        print(f"in torch output but not in jax output: {missing}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
