#!/usr/bin/env python3
"""Step 15: the official JAX code on a Colab TPU v5e-1, the hardware it was tuned for.

Self-contained (installs its own dependencies). Run it over `colab ssh`:
    colab new --tpu v5e1
    colab upload scripts/15_tpu_b0.py /content/15_tpu_b0.py
    colab ssh
    # IMPORTANT: an SSH login shell lacks the TPU_* environment variables Colab sets for its
    # own processes; without them libtpu hangs at startup. Copy them from the Jupyter server:
    eval "$(cat /proc/$(pgrep -f jupyter-server | head -1)/environ | tr '\0' '\n' \
            | grep -iE '^(tpu|xla|jax|pjrt)' | sed 's/^/export /')"
    cd /content && python3 15_tpu_b0.py
    exit
    colab download /content/tpu_b0_results.json results/tpu_b0_results.json
    colab stop
The JSON is also printed at the end, so the terminal output alone is enough.

Measures, with the notebook's TPU settings (splash attention, 128 blocks), default precision:
  1. model step time  : one member, one 6 h step, jitted (same as script 03 on the A10)
  2. ensemble rollout : 32 members x 30 steps (7.5 days, the longest sample file) via the
                        official rollout.chunked_prediction_generator_multiple_runs, timed per
                        member-step (host<->device transfers included, like 13_rollout.py)
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
import numpy as np
import xarray
import xarray_jax
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn

MODEL, SPLIT = "WeatherNextCyclones_Mini", "2024"
BUCKET = "https://storage.googleapis.com/dm_graphcast/"
DATA = ("weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00"
        "_res-1.0_levels-13_steps-30.nc")
WEIGHTS = f"weathernext2/params/{MODEL}_<{SPLIT}.npz"
MEMBERS = 32


def fetch(path: str) -> pathlib.Path:
    local = pathlib.Path("data") / path.replace("<", "lt").replace(" ", "_")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path}", flush=True)
        urllib.request.urlretrieve(BUCKET + urllib.parse.quote(path), local)
    return local


def main() -> None:
    dev = jax.devices()[0]
    print(f"device: {dev.device_kind} x{len(jax.local_devices())}  jax {jax.__version__}  "
          f"backend {jax.default_backend()}", flush=True)
    batch = xarray.load_dataset(fetch(DATA)).compute()
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{MODEL}")
    with open(fetch(WEIGHTS), "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)

    # The notebook's TPU v5e-1 settings (splash attention is the default attention_type).
    tkw = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
    tkw.update({k: 128 for k in ("block_q", "block_kv", "block_kv_compute",
                                 "block_q_dkv", "block_kv_dkv", "block_kv_dkv_compute")})
    print("attention:", tkw.get("attention_type"), flush=True)
    cfg = fgn.PredictorConfig(task=config.task, predictor_constructor=config.predictor_constructor,
                              predictor_kwargs=config.predictor_kwargs,
                              predictor_wrappers=config.predictor_wrappers[:-1])

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        return fgn.construct_predictor(cfg)(inputs, targets_template=targets_template, forcings=forcings)

    run_jit = jax.jit(lambda rng, i, t, f: run_forward.apply(ckpt.params, rng, i, t, f))
    run_pmap = xarray_jax.pmap(run_jit, dim="sample")
    task = dataclasses.asdict(config.task)

    # 1. Model step time (one member, one step).
    i1, t1, f1 = data_utils.extract_inputs_targets_forcings(batch, target_lead_times=slice("6h", "6h"), **task)
    rng = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    jax.block_until_ready(run_jit(rng, i1, t1 * np.nan, f1))
    compile_s = time.perf_counter() - t0
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(run_jit(rng, i1, t1 * np.nan, f1))
        ts.append(time.perf_counter() - t0)
    step_ms = float(np.median(ts) * 1e3)
    print(f"model step: compile {compile_s:.1f} s, median {step_ms:.1f} ms", flush=True)

    # 2. Official ensemble rollout, timed per yielded chunk (1 member x 1 step per chunk on 1 device).
    steps = batch.sizes["time"] - 2
    ie, te, fe = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{steps * 6}h"), **task)
    rngs = np.stack([jax.random.fold_in(rng, m) for m in range(MEMBERS)], axis=0)
    chunk_times, t_last, t_start = [], time.perf_counter(), time.perf_counter()
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=run_pmap, rngs=rngs, inputs=ie, targets_template=te * np.nan, forcings=fe,
            num_steps_per_chunk=1, num_samples=MEMBERS, pmap_devices=jax.local_devices()):
        now = time.perf_counter()
        chunk_times.append(now - t_last)
        t_last = now
    total_s = time.perf_counter() - t_start
    per_chunk = np.array(chunk_times)
    members_per_chunk = len(jax.local_devices())
    steady = per_chunk[steps:]                          # drop the first member (includes compile)
    member_step_s = float(np.median(steady)) / members_per_chunk
    report = {
        "device": dev.device_kind, "num_devices": len(jax.local_devices()), "jax_version": jax.__version__,
        "attention": tkw.get("attention_type"), "model": MODEL,
        "model_step_ms_median": step_ms, "compile_s": compile_s,
        "rollout_members": MEMBERS, "rollout_steps": steps, "rollout_total_s": total_s,
        "rollout_member_step_ms_median": member_step_s * 1e3,
        "seconds_per_15day_member": member_step_s * 60,
        "members_per_hour": 3600 / (member_step_s * 60),
        "usd_per_1000_member_forecasts_at_1.20": member_step_s * 60 / 3600 * 1.20 * 1000,
        "note": "15-day figures extrapolated from 30-step rollout (longest sample file)",
    }
    pathlib.Path("tpu_b0_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
