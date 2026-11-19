"""Step 13: a real 15-day ensemble forecast with the optimized path, timed.

Init: 2024-10-07 00Z (the sample file) - Hurricane Milton is in the Gulf of Mexico.
Each member is the same weights with a fresh 32-number noise draw every step (FGN).
Members are run in batches through the model; the forecast state stays on the GPU.

Records:
  timing    : per-step wall time (model + feature extraction + state update), per batch
  tracks    : per member, per step, the MSLP minimum near the previous storm position
  fields    : ensemble mean + member 0 of MSLP (hPa), 10 m wind speed, 2 m temperature (float16)

    python scripts/13_rollout.py --members 32 --batch 8 --steps 60 --price 1.29
"""
import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
import xarray

spec = importlib.util.spec_from_file_location("b1", pathlib.Path(__file__).parent / "05_b1_torch.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

STEP_S = 6 * 3600
# Milton at init time is in the southern Gulf; search the first minimum in this box.
TRACK_BOX = dict(lat=(15.0, 30.0), lon=(260.0, 280.0))
TRACK_RADIUS_DEG = 5.0       # follow the minimum within this distance of the last position


def track_step(mslp: torch.Tensor, lats, lons, prev: torch.Tensor | None) -> torch.Tensor:
    """mslp [B, lat, lon] -> [B, 3] (lat, lon, min pressure) of the storm center."""
    lat2d, lon2d = torch.meshgrid(lats, lons, indexing="ij")
    if prev is None:
        ok = ((lat2d >= TRACK_BOX["lat"][0]) & (lat2d <= TRACK_BOX["lat"][1]) &
              (lon2d >= TRACK_BOX["lon"][0]) & (lon2d <= TRACK_BOX["lon"][1]))[None]
    else:
        dlon = (lon2d[None] - prev[:, 1, None, None] + 180) % 360 - 180
        dist = torch.sqrt((lat2d[None] - prev[:, 0, None, None]) ** 2 +
                          (dlon * torch.cos(torch.deg2rad(lat2d[None]))) ** 2)
        ok = dist <= TRACK_RADIUS_DEG
    masked = torch.where(ok, mslp, torch.full_like(mslp, float("inf"))).flatten(1)
    idx = masked.argmin(dim=1)
    return torch.stack([lat2d.flatten()[idx], lon2d.flatten()[idx], masked.gather(1, idx[:, None])[:, 0]], 1)


def main(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    processor = b1.WeatherNext2FeatureExtractor.from_pretrained(b1.REPO)
    model = b1.WeatherNext2ForWeatherForecasting.from_pretrained(
        b1.REPO, attn_implementation="eager", dtype=torch.float32).to(dev).eval()
    from mesh_flex_attention import TUNED, install
    install(model, order="hilbert", **TUNED)

    ds = xarray.load_dataset(b1.DATA)
    state0 = {k: v.to(dev) for k, v in b1.load_state(processor, ds).items()}
    t0 = int(ds["seconds_since_epoch"].isel(time=1).values[0])      # last input frame
    lats = torch.tensor(ds.lat.values, device=dev, dtype=torch.float32)
    lons = torch.tensor(ds.lon.values, device=dev, dtype=torch.float32)

    def batch_state(n: int) -> dict:
        return {k: (v if k in processor.static_variables else v.expand(n, *v.shape[1:]).clone())
                for k, v in state0.items()}

    # Warm up + compile with the batch shape we'll use (excluded from timing).
    t_c = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        warm = processor(batch_state(a.batch), seconds_since_epoch=torch.full((a.batch,), t0 + STEP_S),
                         device=dev, noise=torch.zeros(a.batch, processor.noise_channels))
        model(**warm)                                   # builds node order + block mask
    forward = torch.compile(model)
    with torch.inference_mode():
        for _ in range(2):
            forward(**warm)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t_c
    print(f"warmup + compile: {compile_s:.0f} s")

    gen = torch.Generator(device=dev).manual_seed(0)
    nlat, nlon = len(lats), len(lons)
    save = {f: np.zeros((a.steps, 2, nlat, nlon), np.float16) for f in ("mslp", "wind10", "t2m")}
    tracks = np.zeros((a.members, a.steps, 3), np.float32)
    step_times = []                                     # seconds per step, per batch
    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    sums = {f: torch.zeros(a.steps, nlat, nlon, device=dev) for f in save}
    for start in range(0, a.members, a.batch):
        n = min(a.batch, a.members - start)
        state = batch_state(n)
        pos = None
        for s in range(a.steps):
            valid = torch.full((n,), t0 + (s + 1) * STEP_S, dtype=torch.int64)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            with torch.inference_mode():
                noise = torch.randn(n, processor.noise_channels, generator=gen, device=dev)
                inputs = processor(state, seconds_since_epoch=valid, noise=noise, device=dev)
                pred = forward(**inputs).prediction
                phys = processor.postprocess_forecasts(pred, state)
                state = processor.advance_state(state, phys, valid)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - ts)
            # ---- diagnostics (not timed) ----
            mslp = phys["mean_sea_level_pressure"].float() / 100.0      # Pa -> hPa (fits float16)
            wind = torch.hypot(phys["10m_u_component_of_wind"], phys["10m_v_component_of_wind"]).float()
            t2m = phys["2m_temperature"].float()
            pos = track_step(mslp, lats, lons, pos)
            tracks[start:start + n, s] = pos.cpu().numpy()
            for f, x in (("mslp", mslp), ("wind10", wind), ("t2m", t2m)):
                sums[f][s] += x.sum(0)
                if start == 0:
                    save[f][s, 1] = x[0].cpu().numpy().astype(np.float16)
        print(f"members {start}-{start + n - 1}: {np.median(step_times[-a.steps:]) * 1e3:.0f} ms/step "
              f"(batch of {n})", flush=True)
    total_s = time.perf_counter() - t_all
    for f in save:
        save[f][:, 0] = (sums[f] / a.members).cpu().numpy().astype(np.float16)

    step_med = float(np.median(step_times))
    member_step_s = step_med / a.batch                  # one 6 h step for one member
    member_forecast_s = member_step_s * 60              # a 15-day forecast = 60 steps
    report = {
        "device": torch.cuda.get_device_name(0), "members": a.members, "batch": a.batch, "steps": a.steps,
        "variant": "meshflex-hilbert-tuned + torch.compile, TF32",
        "compile_s": compile_s, "total_s_excluding_compile": total_s,
        "step_ms_median_per_batch": step_med * 1e3,
        "seconds_per_15day_member": member_forecast_s,
        "members_per_gpu_hour": 3600 / member_forecast_s,
        "price_per_hour": a.price,
        "usd_per_member_forecast": member_forecast_s / 3600 * a.price,
        "usd_per_1000_member_forecasts": member_forecast_s / 3600 * a.price * 1000,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(json.dumps(report, indent=2))
    pathlib.Path(f"results/rollout_a10_m{a.members}_b{a.batch}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(f"data/rollout_m{a.members}.npz", tracks=tracks, lats=lats.cpu().numpy(),
                        lons=lons.cpu().numpy(), init_seconds=t0, step_s=STEP_S,
                        step_times=np.array(step_times), **save)
    print(f"saved results/rollout_a10_m{a.members}_b{a.batch}.json and data/rollout_m{a.members}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--price", type=float, default=1.29, help="GPU $/hour, for cost per forecast")
    main(ap.parse_args())
