"""Step 5 (B1): the community PyTorch port, timed the same way as B0.

Same model (HF `kashif/weathernext2-mini` main == WeatherNextCyclones_Mini_<2024),
same input file, one ensemble member, one 6-hour step.

Run on the GPU box, inside .venv-torch (after 03 has downloaded the sample data):
    python scripts/05_b1_torch.py --tf32                      # timing, comparable to B0
    python scripts/05_b1_torch.py --noise data/parity_noise.npy --tag parity   # full fp32
    python scripts/05_b1_torch.py --tf32 --attn flex_attention                 # other backends
    python scripts/05_b1_torch.py --tf32 --compile                             # torch.compile
    python scripts/05_b1_torch.py --tf32 --attn sdpa --attn-dtype bf16         # 16-bit attention
    python scripts/05_b1_torch.py --tf32 --mesh-attn hilbert   # full-mesh flex, Hilbert order
    python scripts/05_b1_torch.py --tf32 --profile results/profiles/b1         # + trace
    python scripts/05_b1_torch.py --tf32 --nvtx-probe 8   # mark layer 8's attention, once,
                                                          # as NVTX range "attn_probe" for ncu
    python scripts/05_b1_torch.py --tf32 --compile --nvtx-step   # whole step as NVTX "step_probe"
                                     # (works with torch.compile; per-layer hooks would split the graph)

The trace is a Chrome/Perfetto JSON: drag it into https://ui.perfetto.dev
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
import xarray
from transformers import WeatherNext2FeatureExtractor, WeatherNext2ForWeatherForecasting

REPO = "kashif/weathernext2-mini"
DATA = pathlib.Path("data/weathernext2/dataset/"
                    "source-hres_forecast_init-2024-10-07_00:00:00_res-1.0_levels-13_steps-01.nc")
N_WARMUP, N_TIMED = 3, 20


def load_state(processor, ds: xarray.Dataset) -> dict[str, torch.Tensor]:
    """Physical inputs keyed by variable: the 2 conditioning frames (time 0h, 6h)."""
    state = {}
    for var in processor.input_variables:
        if var not in ds:
            continue  # clock forcings the processor derives from the timestamp
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=slice(0, 2))
        state[var] = torch.from_numpy(np.ascontiguousarray(da.values))
    return state


def main(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = a.tf32   # JAX on Ampere defaults to TF32
    torch.backends.cudnn.allow_tf32 = a.tf32
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"attn={a.attn}  tf32={a.tf32}")

    if a.attn_dtype != "fp32":
        from attn_precision import patch_attention_dtype
        patch_attention_dtype({"bf16": torch.bfloat16, "fp16": torch.float16}[a.attn_dtype])

    processor = WeatherNext2FeatureExtractor.from_pretrained(REPO)
    model = WeatherNext2ForWeatherForecasting.from_pretrained(
        REPO, attn_implementation=a.attn, dtype=torch.float32).to(dev).eval()
    if a.mesh_attn != "none":
        from mesh_flex_attention import TUNED, install
        install(model, order=a.mesh_attn,
                attn_dtype={"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.attn_dtype],
                **(TUNED if a.flex_tuned else {}))

    ds = xarray.load_dataset(DATA)
    state = load_state(processor, ds)
    valid_time = torch.as_tensor(ds["seconds_since_epoch"].isel(time=2).values, dtype=torch.int64)
    noise = torch.from_numpy(np.load(a.noise)).float() if a.noise else None
    gen = None if noise is not None else torch.Generator(device=dev).manual_seed(0)
    inputs = processor(state, seconds_since_epoch=valid_time, noise=noise,
                       generator=gen, device=dev)

    # torch.compile traces the model once and generates fused GPU kernels
    # (via Triton) for chains of elementwise ops. First calls are slow (compiling).
    if a.mesh_attn != "none":
        # Build the node order + block mask once, eagerly (it uses numpy, which
        # torch.compile can't trace), so the compiled graph only sees tensors.
        with torch.inference_mode():
            model(**inputs)
    forward = torch.compile(model) if a.compile else model

    @torch.inference_mode()
    def step():
        return forward(**inputs).prediction

    if a.nvtx_probe is not None:
        # NVTX = named markers in the CUDA timeline. Nsight can filter on them,
        # e.g. ncu --nvtx --nvtx-include "attn_probe/" profiles only kernels
        # launched inside this range. Armed for exactly one call (below).
        probe = {"armed": False}
        attn = model.model.mesh_transformer.layers[a.nvtx_probe].self_attn

        def push(module, args):
            if probe["armed"]:
                torch.cuda.nvtx.range_push("attn_probe")

        def pop(module, args, output):
            if probe["armed"]:
                torch.cuda.nvtx.range_pop()

        attn.register_forward_pre_hook(push)
        attn.register_forward_hook(pop)

    # Like JAX, CUDA work is asynchronous: synchronize() waits for the GPU.
    t0 = time.perf_counter()
    for _ in range(min(N_WARMUP, a.n_timed)):
        pred = step()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    if a.nvtx_step:
        torch.cuda.nvtx.range_push("step_probe")
        step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    if a.nvtx_probe is not None:
        probe["armed"] = True
        step()
        torch.cuda.synchronize()
        probe["armed"] = False

    times = []
    for _ in range(a.n_timed):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times_ms = np.array(times) * 1e3
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"warmup ({min(N_WARMUP, a.n_timed)} calls): {warmup_s:.1f} s")
    print(f"step time: median {np.median(times_ms):.1f} ms "
          f"(min {times_ms.min():.1f}, max {times_ms.max():.1f}) over {a.n_timed} runs")
    print(f"peak GPU memory (allocated): {peak_gb:.2f} GB")

    if a.profile:
        from torch.profiler import ProfilerActivity, profile, record_function
        out_dir = pathlib.Path(a.profile)
        out_dir.mkdir(parents=True, exist_ok=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(3):
                with record_function(f"forecast_step_{i}"):
                    step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out_dir / "trace.json"))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        (out_dir / "top_ops.txt").write_text(table)
        print(table)

    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    attn_name = a.attn if a.mesh_attn == "none" else f"meshflex-{a.mesh_attn}" + ("-tuned" if a.flex_tuned else "")
    tag = f"b1_torch_mini_{gpu}_{attn_name}" + ("_compile" if a.compile else "") + (
        f"_attn{a.attn_dtype}" if a.attn_dtype != "fp32" else "") + ("_tf32" if a.tf32 else "_fp32") + (f"_{a.tag}" if a.tag else "")
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/{tag}.json").write_text(json.dumps({
        "baseline": "B1", "repo": REPO, "device": torch.cuda.get_device_name(0),
        "attn_implementation": attn_name, "attn_dtype": a.attn_dtype, "compile": a.compile, "tf32": a.tf32, "ensemble_members": 1,
        "step_ms_median": float(np.median(times_ms)), "step_ms_all": times_ms.round(2).tolist(),
        "peak_gb": peak_gb, "torch_version": torch.__version__,
        "noise": a.noise or "torch.Generator(seed 0)",
    }, indent=2))
    physical = processor.postprocess_forecasts(pred, state)
    np.savez(f"data/{tag}_prediction.npz", **{k: v.cpu().numpy() for k, v in physical.items()})
    print(f"saved results/{tag}.json and data/{tag}_prediction.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa", "flex_attention"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--n-timed", type=int, default=N_TIMED, help="timed steps (fewer for slow parity runs)")
    ap.add_argument("--mesh-attn", default="none", choices=["none", "native", "hilbert", "morton"],
                    help="replace blocked attention with full-mesh flex in this node order")
    ap.add_argument("--flex-tuned", action="store_true", help="64x64 tiles, 8 warps (scripts/12)")
    ap.add_argument("--attn-dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="precision of the attention computation (default fp32, like the original)")
    ap.add_argument("--noise", metavar="NPY")
    ap.add_argument("--tag", default="")
    ap.add_argument("--profile", metavar="DIR", help="also capture a 3-step trace")
    ap.add_argument("--nvtx-probe", type=int, metavar="LAYER",
                    help="wrap this layer's attention (one call) in NVTX range attn_probe")
    ap.add_argument("--nvtx-step", action="store_true",
                    help="wrap one whole forecast step in NVTX range step_probe")
    main(ap.parse_args())
