#!/usr/bin/env python3
"""Step 16: make the TPU comparison fair - precision check + splash block-size tuning.

1. Parity: one step with the SAME fixed noise vector as the A10 parity runs, at JAX's
   default TPU matmul precision and at "highest" (full fp32). Reports the TPU's own
   precision drift (compare: A10 TF32 drift, worst variable ~1.2e-2) and saves both
   predictions for cross-checking against the A10 fp32 reference with script 06.
2. Tuning: sweeps splash attention block sizes (block_q, block_kv, block_kv_compute),
   timing the model step for each - the TPU counterpart of scripts/12_flex_tuning.py.

Run exactly like script 15 (colab ssh + the TPU env `eval` line, see 15's docstring):
    colab upload scripts/16_tpu_tuning_parity.py /content/16_tpu_tuning_parity.py
    cd /content && python3 16_tpu_tuning_parity.py
    colab download /content/tpu_tuning_parity.json results/tpu_tuning_parity.json
    colab download /content/tpu_parity_highest_prediction.nc data/tpu_parity_highest_prediction.nc
    colab download /content/tpu_parity_default_prediction.nc data/tpu_parity_default_prediction.nc
Precision-matched timing of chosen block sizes only (no parity, ~2 min):
    python3 16_tpu_tuning_parity.py --skip-parity --sweep-precision highest --blocks 128,128,128 512,1024,512
"""
import os
import subprocess
import sys

CHECK = "import jax; print(jax.default_backend(), jax.devices())"


def tpu_ok() -> bool:
    """Does JAX see the TPU in a fresh process? (JAX silently falls back to CPU otherwise.)"""
    try:
        r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=180,
                           env={**os.environ, "JAX_PLATFORMS": "tpu"})
    except subprocess.TimeoutExpired:
        print("jax check: TIMED OUT after 180 s - is another process holding the TPU? "
              "(ps aux | grep python; lsof -w /dev/accel* /dev/vfio/*)", flush=True)
        return False
    lines = (r.stdout or r.stderr or "(no output)").strip().splitlines() or ["(no output)"]
    print("jax check:", lines[-1][:300], flush=True)
    return r.returncode == 0 and r.stdout.startswith("tpu")


if "--installed" not in sys.argv:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    print("before install:", flush=True)
    tpu_ok()
    # Install the official package WITHOUT --upgrade, so pip keeps the image's
    # TPU-enabled jax/libtpu if they already satisfy the requirements.
    subprocess.run(pip + ["--ignore-requires-python",
                          "https://github.com/google-deepmind/weathernext/archive/refs/tags/v0.3.0.zip"], check=True)
    if not tpu_ok():
        # Only now repair jax + libtpu (matching versions from PyPI).
        print("TPU not visible after install; reinstalling jax[tpu]", flush=True)
        subprocess.run(pip + ["-U", "jax[tpu]"], check=True)
        if not tpu_ok():
            subprocess.run([sys.executable, "-m", "pip", "list"], check=False)
            sys.exit("JAX cannot see a TPU (see the jax check lines and pip list above).")
    os.environ["JAX_PLATFORMS"] = "tpu"          # fail loudly instead of falling back to CPU
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--installed"])

import dataclasses
import json
import pathlib
import time
import urllib.parse
import urllib.request

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax
from weathernext.utils import checkpoint, data_utils, fiddle_config_io
from weathernext.weathernext2 import fgn

MODEL, SPLIT = "WeatherNextCyclones_Mini", "2024"
BUCKET = "https://storage.googleapis.com/dm_graphcast/"
DATA = ("weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00"
        "_res-1.0_levels-13_steps-01.nc")       # same file as the A10 parity runs
WEIGHTS = f"weathernext2/params/{MODEL}_<{SPLIT}.npz"
BLOCKS = [  # (block_q, block_kv, block_kv_compute); first = notebook setting
    (128, 128, 128), (256, 256, 256), (256, 256, 128), (256, 512, 256), (512, 512, 512),
    (512, 512, 256), (128, 512, 128), (512, 1024, 512), (1024, 1024, 512),
]


def fetch(path: str) -> pathlib.Path:
    local = pathlib.Path("data") / path.replace("<", "lt").replace(" ", "_")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path}", flush=True)
        urllib.request.urlretrieve(BUCKET + urllib.parse.quote(path), local)
    return local


def fixed_noise_generator(noise: np.ndarray):
    """Same as script 03: the noise vector comes from disk instead of jax.random."""
    def generate(inputs, forcings, default_dtype, noise_var_dict):
        inputs, forcings = inputs.copy(), forcings.copy()
        for var, cfg in noise_var_dict.items():
            source = inputs if cfg["source"] == "input" else forcings
            dims = [d for d, _ in cfg["non_batch_shape"]]
            sizes = [n for _, n in cfg["non_batch_shape"]]
            values = jnp.asarray(noise, dtype=default_dtype).reshape(source.sizes["batch"], *sizes)
            source[var] = xarray_jax.Variable(("batch", *dims), values)
        return inputs, forcings
    return generate


def rel_rms(ref: xarray.Dataset, test: xarray.Dataset) -> dict:
    """Per variable: RMS(test - ref) / std(ref), NaNs ignored (same metric as script 06)."""
    out = {}
    for v in ref.data_vars:
        a, b = ref[v].values.astype(np.float64), test[v].values.astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            out[v] = float(np.sqrt(np.mean((b - a)[ok] ** 2)) / (np.std(a[ok]) or 1.0))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> None:
    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}  jax {jax.__version__}", flush=True)
    batch = xarray.load_dataset(fetch(DATA)).compute()
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{MODEL}")
    with open(fetch(WEIGHTS), "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)
    noise = np.random.default_rng(0).standard_normal((1, 32)).astype(np.float32)  # = data/parity_noise.npy
    config.predictor_kwargs["noise_generator_constructor"] = fixed_noise_generator(noise)
    tkw = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
    cfg = fgn.PredictorConfig(task=config.task, predictor_constructor=config.predictor_constructor,
                              predictor_kwargs=config.predictor_kwargs,
                              predictor_wrappers=config.predictor_wrappers[:-1])
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(config.task))
    template = targets * np.nan
    rng = jax.random.PRNGKey(0)

    def build(blocks):
        """A fresh jitted forward with these splash block sizes (read at trace time)."""
        bq, bkv, bkvc = blocks
        tkw.update(block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
                   block_q_dkv=128, block_kv_dkv=128, block_kv_dkv_compute=128)

        @hk.transform
        def run_forward(i, t, f):
            return fgn.construct_predictor(cfg)(i, targets_template=t, forcings=f)
        return jax.jit(lambda r, i, t, f: run_forward.apply(ckpt.params, r, i, t, f))

    # 1. Precision: default (what script 15 timed) vs highest (full fp32 matmuls).
    preds, drift, worst = {}, {}, ("skipped", float("nan"))
    if args.skip_parity:
        print("parity check skipped", flush=True)
    for prec in (() if args.skip_parity else ("default", "highest")):
        with jax.default_matmul_precision(prec):
            fn = build(BLOCKS[0])
            preds[prec] = jax.device_get(jax.block_until_ready(fn(rng, inputs, template, forcings)))
        preds[prec].to_netcdf(f"tpu_parity_{prec}_prediction.nc")
    if preds:
        drift = rel_rms(preds["highest"], preds["default"])
        worst = next(iter(drift.items()))
        print(f"TPU default vs highest precision, worst rel RMS: {worst[0]} {worst[1]:.2e}", flush=True)
        for v in list(drift)[:12]:
            print(f"  {v:36s} {drift[v]:.2e}", flush=True)

    # 2. Splash block-size sweep (default precision unless --sweep-precision highest).
    sweep = []
    for blocks in (BLOCKS if not args.blocks else [tuple(int(x) for x in b.split(",")) for b in args.blocks]):
        try:
            with jax.default_matmul_precision(args.sweep_precision):
                fn = build(blocks)
                t0 = time.perf_counter()
                jax.block_until_ready(fn(rng, inputs, template, forcings))
                compile_s = time.perf_counter() - t0
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(rng, inputs, template, forcings))
                    ts.append(time.perf_counter() - t0)
            ms = float(np.median(ts) * 1e3)
            sweep.append({"blocks": blocks, "step_ms": ms, "compile_s": compile_s,
                          "precision": args.sweep_precision})
            print(f"  splash blocks q/kv/kv_compute {blocks}: {ms:7.1f} ms  (compile {compile_s:.0f} s)",
                  flush=True)
        except Exception as ex:                      # invalid combination / out of memory
            sweep.append({"blocks": blocks, "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
            print(f"  splash blocks {blocks}: failed ({type(ex).__name__})", flush=True)

    ok = [s for s in sweep if "step_ms" in s]
    best = min(ok, key=lambda s: s["step_ms"]) if ok else None
    report = {"device": dev.device_kind, "jax_version": jax.__version__,
              "precision_drift_default_vs_highest": drift, "splash_sweep": sweep, "best": best}
    out = f"tpu_tuning_parity{'_' + args.sweep_precision if args.sweep_precision != 'default' else ''}.json"
    pathlib.Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"worst_drift": worst, "best": best,
                      "baseline_128": ok[0] if ok and ok[0]["blocks"] == BLOCKS[0] else None}, indent=2))
    print(f"saved {out}" + ("" if args.skip_parity else
                             ", tpu_parity_default_prediction.nc, tpu_parity_highest_prediction.nc"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-precision", default="default", choices=["default", "high", "highest"])
    ap.add_argument("--blocks", nargs="*", help="e.g. 128,128,128 512,1024,512 (default: full sweep)")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--installed", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    main()
