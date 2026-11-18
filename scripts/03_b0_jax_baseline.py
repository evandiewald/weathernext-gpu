"""Step 3 (B0): time one forecast step of the official JAX model on GPU.

What this measures, for ONE ensemble member and ONE 6-hour step:
  * compile time   - first call: JAX traces the Python and XLA compiles it
  * step time      - median over repeated calls of the already-compiled function
  * peak GPU memory

Run on the GPU box, inside .venv-jax:
    python scripts/03_b0_jax_baseline.py
    python scripts/03_b0_jax_baseline.py --profile results/profiles/b0   # + trace
    # parity reference: fixed noise vector + full-fp32 matmuls (no TF32)
    python scripts/03_b0_jax_baseline.py --noise data/parity_noise.npy \
        --matmul-precision highest --tag parity

The trace lands in results/profiles/b0/plugins/profile/<timestamp>/:
  *.trace.json.gz -> drag into https://ui.perfetto.dev  (timeline)
  *.xplane.pb     -> `xprof --logdir results/profiles/b0` (op tables)
"""
import argparse
import os

# JAX grabs 75% of GPU memory up front by default, which would make "peak
# memory" meaningless. Turn that off BEFORE importing jax.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

MODEL = "WeatherNextCyclones_Mini"   # the 1-degree model from the demo notebook
SPLIT = "2024"
DATA = ("weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00"
        "_res-1.0_levels-13_steps-01.nc")       # 2 input frames + 1 target
WEIGHTS = f"weathernext2/params/{MODEL}_<{SPLIT}.npz"
BUCKET = "https://storage.googleapis.com/dm_graphcast/"
CACHE = pathlib.Path("data")
N_TIMED = 20


def fetch(path: str) -> pathlib.Path:
    """Download a file from the public GCS bucket once, then reuse it."""
    local = CACHE / path.replace("<", "lt").replace(" ", "_")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path}")
        urllib.request.urlretrieve(BUCKET + urllib.parse.quote(path), local)
    return local


def fixed_noise_generator(noise: np.ndarray):
    """Drop-in for fgn.gaussian_noise_generator that returns a given vector.

    The real one draws jax.random.normal(...). JAX and PyTorch RNGs differ, so
    for the parity check both sides read the same vector from disk instead.
    """
    def generate(inputs, forcings, default_dtype, noise_var_dict):
        inputs, forcings = inputs.copy(), forcings.copy()
        for var, cfg in noise_var_dict.items():
            source = inputs if cfg["source"] == "input" else forcings
            dims = [d for d, _ in cfg["non_batch_shape"]]
            sizes = [n for _, n in cfg["non_batch_shape"]]
            values = jnp.asarray(noise, dtype=default_dtype).reshape(
                source.sizes["batch"], *sizes)
            source[var] = xarray_jax.Variable(("batch", *dims), values)
        return inputs, forcings
    return generate


def main(profile_dir: str | None, noise_path: str | None, precision: str, tag: str) -> None:
    if precision == "highest":
        # Full fp32 matmuls. Default on Ampere GPUs is TF32 (10-bit mantissa).
        jax.config.update("jax_default_matmul_precision", "highest")
    gpu = jax.devices()[0]
    print(f"device: {gpu.device_kind}  (backend={jax.default_backend()})")

    # 1. Load data, config, and weights (same as the demo notebook).
    batch = xarray.load_dataset(fetch(DATA))
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{MODEL}")
    with open(fetch(WEIGHTS), "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(config.task))

    # 2. The one-line GPU switch from the notebook: splash (TPU-only kernel)
    #    -> triblockdiag (plain einsums). This is the thing we're baselining.
    tkw = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords[
        "transformer_kwargs"]
    tkw["attention_type"] = "triblockdiag_mha"
    print("transformer:", {k: tkw[k] for k in
                           ("attention_type", "attention_k_hop", "d_model",
                            "num_heads", "num_layers")})

    if noise_path:
        config.predictor_kwargs["noise_generator_constructor"] = fixed_noise_generator(
            np.load(noise_path))
        print(f"using fixed noise from {noise_path}")

    # Drop the ensemble wrapper: we run a single member.
    cfg = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1])

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        predictor = fgn.construct_predictor(cfg)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    step = jax.jit(lambda rng, i, t, f: run_forward.apply(ckpt.params, rng, i, t, f))
    rng = jax.random.PRNGKey(0)
    template = targets * np.nan

    # 3. Compile + first run. JAX calls are ASYNC: they return immediately and
    #    the GPU works in the background. block_until_ready() waits for the
    #    result - without it we'd be timing how fast Python can queue work.
    t0 = time.perf_counter()
    out = jax.block_until_ready(step(rng, inputs, template, forcings))
    compile_s = time.perf_counter() - t0
    print(f"first call (trace + compile + run): {compile_s:.1f} s")

    # 4. Steady-state timing.
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        jax.block_until_ready(step(rng, inputs, template, forcings))
        times.append(time.perf_counter() - t0)
    times_ms = np.array(times) * 1e3
    stats = gpu.memory_stats() or {}
    peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
    print(f"step time: median {np.median(times_ms):.1f} ms "
          f"(min {times_ms.min():.1f}, max {times_ms.max():.1f}) over {N_TIMED} runs")
    print(f"peak GPU memory: {peak_gb:.2f} GB")

    # 4b. Optional: record a few steps with the profiler. Separate from the
    #     timing loop because profiling adds overhead.
    if profile_dir:
        with jax.profiler.trace(profile_dir, create_perfetto_trace=True):
            for i in range(3):
                with jax.profiler.TraceAnnotation(f"forecast_step_{i}"):
                    jax.block_until_ready(step(rng, inputs, template, forcings))
        print(f"profile written under {profile_dir}")

    # 5. Save numbers + the prediction (ground truth for the PyTorch parity check).
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)
    tag = f"b0_jax_{MODEL}_{gpu.device_kind.replace(' ', '_')}" + (f"_{tag}" if tag else "")
    (out_dir / f"{tag}.json").write_text(json.dumps({
        "baseline": "B0", "model": MODEL, "device": gpu.device_kind,
        "attention_type": tkw["attention_type"], "ensemble_members": 1,
        "compile_s": compile_s, "step_ms_median": float(np.median(times_ms)),
        "step_ms_all": times_ms.round(2).tolist(), "peak_gb": peak_gb,
        "jax_version": jax.__version__, "matmul_precision": precision,
        "noise": noise_path or "jax.random (seed 0)",
    }, indent=2))
    pred = jax.device_get(out)
    pred.to_netcdf(CACHE / f"{tag}_prediction.nc")
    print(f"saved results/{tag}.json and data/{tag}_prediction.nc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", metavar="DIR", help="also capture a 3-step trace")
    ap.add_argument("--noise", metavar="NPY", help="fixed (1, 32) noise vector")
    ap.add_argument("--matmul-precision", default="default", choices=["default", "highest"])
    ap.add_argument("--tag", default="", help="suffix for output file names")
    a = ap.parse_args()
    main(a.profile, a.noise, a.matmul_precision, a.tag)
