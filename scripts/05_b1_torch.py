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

Full 0.25-degree model (WeatherNextCyclones_<2024_model1: 40,962 mesh nodes, 32-hop, d768,
24 layers, 1,038,240 grid points). Downloads the 1.3 GB input file + ~2.2 GB of weights/mask:
    python scripts/05_b1_torch.py --model full --tf32 --mesh-attn hilbert --flex-tuned \
        --n-timed 5 --memory-snapshot results/profiles/full_snapshot.pickle
    python scripts/05_b1_torch.py --model full --tf32 --mesh-attn hilbert --flex-tuned \
        --chunk 65536 --n-timed 5 --memory-snapshot results/profiles/full_chunked_snapshot.pickle
An out-of-memory failure is a result too: it is written to results/<tag>.json (status "oom")
and, with --memory-snapshot, the snapshot shows what was alive. View snapshots by dragging the
.pickle into https://pytorch.org/memory_viz

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

# HF port checkpoints. mini == WeatherNextCyclones_Mini_<2024 (what B0 runs);
# full == WeatherNextCyclones_<2024_model1 (README of kashif/weathernext-cyclones).
MODELS = {
    "mini": dict(repo="kashif/weathernext2-mini", kwargs={}, res="1.0"),
    "full": dict(repo="kashif/weathernext-cyclones", kwargs={"revision": "<2024", "subfolder": "model1"},
                 res="0.25"),
}
BUCKET = "https://storage.googleapis.com/dm_graphcast/"
DATA_NAME = "weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-{res}_levels-13_steps-01.nc"


def fetch_data(res: str) -> pathlib.Path:
    """Same file and local name as script 03's fetch() ("<" and spaces replaced)."""
    import urllib.parse
    import urllib.request
    name = DATA_NAME.format(res=res)
    local = pathlib.Path("data") / name.replace("<", "lt").replace(" ", "_")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(BUCKET + urllib.parse.quote(name), local)
    return local
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
          f"model={a.model}  attn={a.attn}  tf32={a.tf32}  chunk={a.chunk}")
    spec = MODELS[a.model]
    if a.memory_snapshot:
        # Records every allocation (with Python stack) from here on; dumped at the end or on OOM.
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    if a.attn_dtype != "fp32":
        from attn_precision import patch_attention_dtype
        patch_attention_dtype({"bf16": torch.bfloat16, "fp16": torch.float16}[a.attn_dtype])

    processor = WeatherNext2FeatureExtractor.from_pretrained(spec["repo"], **spec["kwargs"])
    model = WeatherNext2ForWeatherForecasting.from_pretrained(
        spec["repo"], **spec["kwargs"], attn_implementation=a.attn, dtype=torch.float32).to(dev).eval()
    if a.chunk:
        from chunked_graph import install_chunked
        install_chunked(model, chunk=a.chunk)
    if a.mesh_attn != "none":
        from mesh_flex_attention import TUNED, install
        install(model, order=a.mesh_attn,
                attn_dtype={"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.attn_dtype],
                **(TUNED if a.flex_tuned else {}))

    ds = xarray.load_dataset(fetch_data(spec["res"]))
    state = load_state(processor, ds)
    valid_time = torch.as_tensor(ds["seconds_since_epoch"].isel(time=2).values, dtype=torch.int64)
    noise = torch.from_numpy(np.load(a.noise)).float() if a.noise else None
    gen = None if noise is not None else torch.Generator(device=dev).manual_seed(0)
    inputs = processor(state, seconds_since_epoch=valid_time, noise=noise,
                       generator=gen, device=dev)

    gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    attn_name = a.attn if a.mesh_attn == "none" else f"meshflex-{a.mesh_attn}" + ("-tuned" if a.flex_tuned else "")
    tag = f"b1_torch_{a.model}_{gpu}_{attn_name}" + ("_compile" if a.compile else "") + (
        f"_attn{a.attn_dtype}" if a.attn_dtype != "fp32" else "") + ("_tf32" if a.tf32 else "_fp32") + (
        f"_chunk{a.chunk}" if a.chunk else "") + (f"_{a.tag}" if a.tag else "")
    meta = {"baseline": "B1", "model": a.model, "repo": spec["repo"], **spec["kwargs"],
            "device": torch.cuda.get_device_name(0), "attn_implementation": attn_name,
            "attn_dtype": a.attn_dtype, "compile": a.compile, "tf32": a.tf32, "chunk": a.chunk,
            "ensemble_members": 1, "torch_version": torch.__version__,
            "noise": a.noise or "torch.Generator(seed 0)"}
    pathlib.Path("results").mkdir(exist_ok=True)
    try:
        timing = run(a, model, inputs, dev)
    except torch.OutOfMemoryError as ex:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        msg = str(ex).splitlines()[0]
        print(f"OUT OF MEMORY (peak allocated before failing: {peak_gb:.2f} GB): {msg}")
        if a.memory_snapshot:
            torch.cuda.memory._dump_snapshot(a.memory_snapshot)
            print(f"memory snapshot: {a.memory_snapshot}")
        pathlib.Path(f"results/{tag}.json").write_text(json.dumps(
            {**meta, "status": "oom", "peak_gb_before_oom": peak_gb, "error": msg,
             "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}, indent=2))
        print(f"saved results/{tag}.json")
        raise SystemExit(1)
    pred, times_ms, peak_gb, warmup_peak_gb = timing
    if a.memory_snapshot:
        torch.cuda.memory._dump_snapshot(a.memory_snapshot)
        print(f"memory snapshot: {a.memory_snapshot}")

    pathlib.Path(f"results/{tag}.json").write_text(json.dumps({
        **meta, "status": "ok",
        "step_ms_median": float(np.median(times_ms)), "step_ms_all": times_ms.round(2).tolist(),
        "peak_gb": peak_gb, "warmup_peak_gb": warmup_peak_gb,
    }, indent=2))
    physical = processor.postprocess_forecasts(pred, state)
    np.savez(f"data/{tag}_prediction.npz", **{k: v.cpu().numpy() for k, v in physical.items()})
    print(f"saved results/{tag}.json and data/{tag}_prediction.npz")


def run(a, model, inputs, dev):
    """Warmup, optional probes, timing. Returns (prediction, step times in ms, peak GB)."""
    if a.mesh_attn != "none":
        # Build the node order + block mask once, eagerly (it uses numpy, which
        # torch.compile can't trace), so the compiled graph only sees tensors.
        with torch.inference_mode():
            model(**inputs)
        # The port's blocked mask is only read on that first call (the full-mesh
        # path caches its own N x N mask); drop it so it doesn't count toward the
        # peak (1.27 GB at 0.25 deg).
        inputs["attention_mask"] = inputs["attention_mask"][:1, :, :1, :1].clone()
        torch.cuda.empty_cache()
    # torch.compile traces the model once and generates fused GPU kernels
    # (via Triton) for chains of elementwise ops. First calls are slow (compiling).
    forward = torch.compile(model) if a.compile else model

    @torch.inference_mode()
    def step():
        return forward(**inputs).prediction

    if a.nvtx_probe is not None:
        # NVTX = named markers in the CUDA timeline. Nsight can filter on them,
        # e.g. ncu --nvtx --nvtx-include "attn_probe/" profiles only kernels
        # launched inside this range. Armed for exactly one call (below).
        if a.compile:
            raise SystemExit("--nvtx-probe needs eager mode (hooks split compiled graphs); use --nvtx-step")
        probe = {"armed": False}
        enc = model.model.mesh_transformer
        enc.probe_layer = a.nvtx_probe          # used by the full-mesh flex path
        attn = enc.layers[a.nvtx_probe].self_attn

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
    # Peak including one-time setup (mask build, compile), then reset so peak_gb
    # below is the steady-state forecast step.
    warmup_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    if a.nvtx_step:
        torch.cuda.nvtx.range_push("step_probe")
        step()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    if a.nvtx_probe is not None:
        probe["armed"] = enc.probe_armed = True
        step()
        torch.cuda.synchronize()
        probe["armed"] = enc.probe_armed = False

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
    print(f"peak GPU memory (allocated): {peak_gb:.2f} GB  (incl. one-time setup: {warmup_peak_gb:.2f} GB)")

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

    return pred, times_ms, peak_gb, warmup_peak_gb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mini", choices=list(MODELS), help="mini (1 deg) or full (0.25 deg)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="rows per chunk for grid-scale layers (scripts/chunked_graph.py); 0 = off")
    ap.add_argument("--memory-snapshot", metavar="PICKLE",
                    help="record allocations; dump at the end or on OOM (view at pytorch.org/memory_viz)")
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
