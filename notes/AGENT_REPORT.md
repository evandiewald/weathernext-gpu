# WeatherNext 2 inference profiling/optimization: agent handoff report

State as of 2026-11-19 (updated after a review session the same day: precision framing in
sec. 6, TPU reinterpretation in sec. 8, full-model memory work in sec. 8b). Audience: an agent with this repo. Numbers are from `results/`; script numbers
refer to `scripts/NN_*.py` (each has a usage docstring). "Mini" = `WeatherNextCyclones_Mini_<2024`
(1.0 deg grid, 10,242-node mesh = icosahedron refined 5x, 16 layers, d_model 512, 4 heads x 128,
16-hop attention window). Everything below is Mini unless stated. All GPU numbers: Lambda `gpu_1x_a10`
(A10 24 GB, sm_86); one ensemble member, one 6 h step unless stated.

## 1. Headline

| Path (A10 unless noted) | step | notes |
|---|---|---|
| B0: official JAX, GPU fallback `triblockdiag_mha`, TF32 default | 424 ms | 2.33 GB peak |
| B1: HF PyTorch port, eager, TF32 | 599 ms | 3.60 GB; parity vs B0-fp32 1.7e-5 |
| **Best exact path**: full-mesh Flex, Hilbert order, tuned tiles, `torch.compile`, TF32 | **132 ms** | 1.83 GB; fp32 parity 1.7e-5 |
| Best bf16-attention path: SDPA + compile, attn bf16 | 182.5 ms | precision trade (sec. 6) |
| TPU v5e-1 (Colab), official JAX splash, notebook override (all blocks 128) | 191 ms | default TPU precision; NOT Google's tuned config (sec. 8) |
| TPU v5e-1, splash blocks = shipped config 1024/1024/512 | **61.0 ms** | the real TPU baseline; default precision (~bf16 matmuls) |
| TPU v5e-1, best of our sweep 512/1024/512 | 59.5 ms | only +2.6% over shipped; outputs not yet checked vs 128 blocks |

15-day 32-member rollout on A10 (best exact path): 8.16 s/member, 441 members/GPU-h, $2.92 / 1000
member-forecasts at $1.29/h. TPU (notebook 128 blocks): 10.9 s/member, $3.63/1000 at $1.20/chip-h. Those TPU
cost numbers use the notebook override, not Google's config; redo with shipped blocks (~3.6 s/member
extrapolated, not measured).

## 2. Environment / how to reproduce

- A10 box: `bash scripts/setup_instance.sh`, then `sudo reboot` (required: sets
  `NVreg_RestrictProfilingToAdminUsers=0` so non-root ncu / Nsight GUI can read counters; also runs
  `update-initramfs`). Then regenerate reference data (commands in setup script header):
  B0 run, `data/parity_noise.npy` (`default_rng(0).standard_normal((1,32))`), B0 parity run
  (`--noise data/parity_noise.npy --matmul-precision highest --tag parity`).
- Two venvs via `uv` + Python 3.12 (system 3.10 too old for current JAX): `.venv-jax`
  (`requirements-jax.txt`, weathernext pinned v0.3.0) and `.venv-torch` (`requirements-torch.txt`,
  HF port from `kashif/transformers@add-weathernext2`, PR huggingface/transformers#47874). Need `h5py`
  (jax env) and `h5netcdf` (torch env) for xarray.
- `third_party/weathernext` (git-ignored) = official repo clone; scripts 01, 10-12 and
  `mesh_flex_attention.py` import its `icosahedral_mesh.py` directly (numpy/scipy only).
- Data/weights: public GCS `https://storage.googleapis.com/dm_graphcast/weathernext2/...` over plain
  HTTPS (URL-quote the space and `<`). Sample files exist for steps 01/04/12/20/30 only (max 7.5 days of
  forcings for the official JAX rollout; HF port derives forcings from timestamps so it can run 60 steps).
- ncu: Lambda apt `nsight-compute`, binary `/usr/lib/nsight-compute/ncu`.
- Lambda instances can't be stopped, only terminated (disk wiped). Repo is the source of truth.
- TPU: Colab CLI (`google-colab-cli`). `colab new --tpu v5e1`, `colab upload`, `colab ssh`.
  **Gotcha:** an SSH login shell lacks the `TPU_*` env vars Colab sets for its own processes; libtpu
  then hangs at init (silently, no error). Fix (also in script 15 docstring):
  `eval "$(cat /proc/$(pgrep -f jupyter-server|head -1)/environ | tr '\0' '\n' | grep -iE '^(tpu|xla|jax|pjrt)' | sed 's/^/export /')"`.
  Do NOT follow the notebook's `pip uninstall libtpu; pip install jax[tpu] -f libtpu_releases` on
  current images (py3.13, jax 0.11.2, libtpu 0.0.50): it can break the TPU backend; JAX then falls back
  to CPU and splash fails with "Only interpret mode is supported on CPU backend". Scripts 15/16 check
  the backend in a subprocess first and run with `JAX_PLATFORMS=tpu`.

## 3. Model facts that matter for performance

- Encode-process-decode. Processor = 16 transformer layers over mesh nodes; attention restricted to
  k-hop neighbourhood (16 hops Mini: 799 neighbours/node, 7.80% density; full model 32 hops: 3,101,
  7.57%).
- Official code renumbers nodes with RCM (`icosahedral_mesh.get_permutation_to_banded`). TPU path:
  `splash_mha` (Pallas, TPU-only, block-sparse). GPU path (`triblockdiag_mha`, set in demo notebook):
  blocks of size = RCM bandwidth (Mini 2,577; full 10,273), each block attends to [prev, own, next]
  block via einsums. Full model: scores per layer = 4 blocks x 6 heads x 10,273 x 30,819 = 7.6e9
  elements = **~30 GB fp32** (earlier versions said "7.6 GB": elements, not bytes) -> why README says
  H100, and why B0 cannot run the full model on a 24 GB A10.
- Splash block sizes: the RELEASED configs already use large blocks (Mini fwd 1024/1024/512, dkv
  512/2048/512; full WN2 1024/512/256; GenCast defaults ~1024). The demo notebook overrides all six to
  128 with the comment "Overriding the following block sizes enable running the model on a TPU v5e-1 ...
  some tuning is recommended." (it also computes gradients). The notebook also calls the GPU
  triblockdiag path "slower and significantly more memory hungry": B0 is the official fallback, not a
  tuned GPU path.
- **No released config uses the bf16 cast wrapper**: compiled program has zero bf16 tensors; GPU runs
  fp32 matmuls as TF32 by default. `upcast_attn_to_fp32=True` in configs (helper comments say the code
  was designed for bf16 activations with fp32 softmax).
- HF port mirrors triblockdiag (concatenates K/V of 3 blocks -> scores `[blocks*heads, 2577, 7731]`
  fp32 = 1.27 GB), upcasts Q/K/V to fp32, supports `attn_implementation` eager/sdpa/flex_attention.
  Port node index i == icosahedral vertex `rcm_perm[i]` (verified). Padding nodes never take part in
  attention (verified in `mesh_flex_attention.full_mask_from_blocked`).

## 4. Methodology / conventions

- Timing: median of 20 steps after warmup, `block_until_ready` / `torch.cuda.synchronize`.
  `--n-timed` reduces it (Flex in strict fp32 is ~4-10 s/step).
- Parity (script 06): fixed noise vector on both sides (JAX side swaps `noise_generator_constructor`,
  script 03 `--noise`; torch side `noise=`), metric = RMS(diff)/std(ref field) per variable, vs the
  JAX fp32 (`highest`) prediction. Exactness check = full fp32 both sides (expect ~1e-5). TF32 runs
  show ~1e-2 worst-variable drift regardless of method; control (different noise) gives 0.2-1.2.
  Report worst variable AND specific ones (2m T, T, MSLP, 10m u, vertical velocity).
  `cyclone_exists_gaussian_unit_mode` has no normalization stats; noisy, don't headline it.
- JAX traces: disable command buffers (`XLA_FLAGS=--xla_gpu_enable_command_buffer=`) or kernels lose
  `hlo_op`; map kernels to model code via HLO dump (`--xla_dump_to ... --xla_dump_hlo_as_text`) +
  script 04.
- ncu locks clocks: compute-bound kernel durations inflate (QK^T 1.23 -> 1.92 ms), memory-bound ones
  don't. Bytes are exact. Script 07 parses ncu CSV (bytes/time per kernel family).
- NVTX probes in script 05: `--nvtx-probe L` -> range `attn_probe` around layer L attention, one call
  (eager only; per-layer hooks split `torch.compile` graphs); `--nvtx-step` -> `step_probe` around a
  whole step (works with compile). ncu: `--nvtx --nvtx-include "attn_probe/"` (trailing slash = push/pop
  range). In the Nsight GUI, add the NVTX rule via the "..." dialog and quote regex kernel filters
  (the GUI passes them unquoted through bash; `|` becomes a pipe). Template kernels (cuBLAS/CUTLASS)
  need `--kernel-name-base demangled` to match on `s1688gemm` etc.

## 5. Findings, in order (what the profiles showed -> what we did)

1. B0 breakdown (script 04, 424 ms): mask+softmax 198 ms (47%), QK^T 62, AV 51, FFW 34, Q/K/V/out proj
   14, enc/dec 26, other 36. Encoder/decoder is NOT the problem on Mini.
2. ncu on B0 softmax kernels: max/sum/divide at 96/97/92% of DRAM peak, 15-26% SM -> memory-bound at
   ceiling; only fix is fewer bytes (fuse / don't materialize scores).
3. B1 eager is slower than B0 (599 vs 424) because scale/mask/softmax are separate passes over the
   1.27 GB score tensor. Layer-8 probe: 23 kernels, 36.8 ms, 15.1 GB DRAM vs ~0.2 GB actually needed;
   95% of bytes are the score tensor.
4. `torch.compile` (eager attention): 346 ms, beats B0. Reason: Inductor fuses scale+mask+add+softmax
   into ONE kernel/layer (6.3 ms, 3.4 GB) vs XLA's three (10.4 ms, 5.9 GB) - largely due to the port's
   concatenated-block layout. Not "PyTorch > JAX"; untested: rewriting JAX triblockdiag to concat blocks.
5. Bytes/step (one step, ncu): eager 283 GB, compile 154 GB, port-Flex 205 GB (154 GB inside the flex
   kernel), SDPA 105 GB but SDPA's fused kernel only ~130 GB/s: fp32 mem-efficient attention can't use
   tensor cores (compute-bound on CUDA cores). Flash backend unavailable in fp32.
6. Port Flex underperformance root cause (scripts 08/09): BlockMask works (45% tiles skipped, 0% full);
   cost is mask_mod reading the dense bool mask per element (~16K scattered loads/tile).
   One layer, port layout: A exact 17.4 ms (fp32/TF32) / 10.2 ms (bf16); B same tiles no mask 4.8/1.2;
   F arithmetic-only mask 5.4/1.6; C all tiles 7.8/2.3; SDPA bool mask 27.1/12.0. Bit-packing the mask
   (8x fewer bytes) only ~20% faster -> it's load count, not bytes.
7. Full-mesh Flex (one N x N block-sparse call, `mesh_flex_attention.py`) makes lookups much cheaper:
   one layer RCM 10.5 / Hilbert 8.2 ms TF32 (3.8 / 3.1 bf16).
8. Node ordering (script 10, full N x N mask): tiles computed at 128x128: RCM 2741 (waste 5.49x),
   lat/lon 3097 (6.20x), Morton 2251 (4.51x), **Hilbert 1955 (3.91x)**; at 64x64: RCM 5.31x, Morton
   3.25x, Hilbert 2.91x. RCM tiles are thin strips (BFS level sets) so smaller tiles don't help RCM.
   Full tiles stay rare (Hilbert: 5 @128, 144 @64), so mask_mod remains on almost all tiles.
9. In-model: meshflex native 286 ms (+compile 250), Hilbert 245 ms (+compile 206.5). fp32 parity
   1.9e-5.
10. ncu on Hilbert flex kernel: latency-bound, occupancy 8.3% (1 block x 4 warps/SM, smem-limited at
    128x128 fp32 tiles), 324 blocks / 72 SMs = 4.5 waves, L1 hit 4%.
11. Flex kernel tuning (script 12; `kernel_options` + BlockMask BLOCK_SIZE): best mask 64, BLOCK_M=64,
    BLOCK_N=64, num_warps=8 (stages 1-3 equal): **3.17 ms/layer vs 8.2 default**. All 128x128 configs
    OOM shared memory in fp32. Stored as `mesh_flex_attention.TUNED`; script 05 `--flex-tuned`.
    In-model: 167 ms, +compile 132 ms, fp32 parity 1.7e-5.
12. ncu on tuned flex kernel (layer 8 probe: 11 kernels, 6.6 ms, 0.9 GB vs 23 / 36.8 ms / 15.1 GB):
    2.90 ms, DRAM 30%, SM 35%, tensor pipe busiest, occupancy 16.7%; limiters now: **255 regs/thread
    with local-memory spills (36K requests)**, block limit 1 by regs and smem; shared-memory bank
    conflicts (ncu est. 14%); uncoalesced global (mask) only est. 2.4%. Attention now ~46 of ~130 ms.

## 6. Precision track (separate from the exact path)

- `--attn-dtype bf16` (`attn_precision.py`) patches the port's attention to cast Q/K/V to bf16 instead
  of fp32 (rest of model unchanged). Results: SDPA 252 ms, SDPA+compile 182.5 ms (1.73 GB), Flex 328,
  Flex+compile 275. Eager+compile bf16 fails (fp32 additive mask + bf16 scores dtype mismatch).
- Drift (SDPA+compile bf16 vs JAX fp32): worst vertical velocity 3.1e-2, 10m u 6.3e-3, 2m T 9.4e-4
  (max 1.4 K), MSLP 8.9e-4, T 4.8e-4 -> ~2-3x TF32 drift, ~30-40x below member-to-member spread.
  Only a one-step check; multi-step skill vs observations NOT done. bf16 not combined with
  meshflex-Hilbert-tuned yet (needs its own script 12 sweep).

### Precision framing for the A10 headline (review)
- B0 default and the 132 ms path both use TF32 matmuls with fp32 softmax (B0 default 420 ms vs
  `highest` 581 ms confirms JAX default = TF32). The speedup is from bytes (score tensor never built),
  not precision; the bf16 path (182.5) is slower than the exact one (132).
- Caveat: the win needs TF32. Strict fp32 both sides: B0 581 ms vs best path ~4,000 ms (Flex Triton
  kernel only uses tensor cores via TF32). Article wording: "same precision as the official GPU
  default, verified against fp32", not "faster in fp32".
- Gap: TF32 drift of the exact timed configs (132 ms path; B0 default) was never measured; only
  strict-fp32 parity and eager-TF32 (~1e-2). One parity run each.
- Attribution: port eager 599 -> compile 346 is Inductor fusion (layout-specific, see 5.4);
  346 -> 132 is the algorithm (block-sparse = what splash does on TPU). Frame as "GPU fallback
  replaced with the TPU path's idea", not "PyTorch beats JAX".
- Memory: peak 2.32 (B0) -> 1.83 GB (best) = -21% only; traffic is the real change (layer-8 attention
  15.1 -> 0.9 GB vs port eager; whole step 283 GB eager -> 77 GB Hilbert+compile, tuned not measured).

## 7. Things that did not work / dead ends

- SDPA in fp32: no tensor cores; fused but slow. SDPA+compile (363) < eager+compile (346).
- Port-layout Flex: dominated by dense-mask lookups (sec. 5.6). Flex with fp32 and TF32 disabled:
  ~9 s/step (no tensor cores in its Triton kernel) - use `--n-timed 1` for strict-fp32 parity runs.
- Bit-packed mask_mod: ~20% faster only, and wrong in one tile (block 0, queries 0-127) under flex
  although it unpacks correctly on host. Not debugged; removed from script 09.
- Batching ensemble members on A10: no throughput gain (1 member already saturates; batch 8 =
  ~130 ms/member-step), batch 16 OOM.
- `torch.compile` + numpy inside forward (node order build) -> dynamo guard assertion; fixed by one
  eager forward before compiling (script 05/13 do this).
- Flex 128x128 tiles in fp32: shared-memory OOM in every stage/warp combination.
- Initial claim "model runs in bf16" was wrong (sec. 3).

## 8. TPU comparison (Colab TPU v5e-1, official JAX 0.11.2, splash)

- Script 15 (notebook blocks 128): model step 191 ms, compile 43 s; official rollout utility
  32 members x 30 steps: 181 ms/member-step (one member per chunk on 1 device; host<->device per step
  included).
- Script 16 block sweep (default precision, ms/step; dkv blocks fixed at 128 = irrelevant for inference): 128/128/128 186.7; 256/256/256 85.5; 256/512/256
  69.0; 512/512/512 63.1; 512/512/256 63.4; 128/512/128 89.6; **512/1024/512 59.5**; 1024/1024/512 61.0.
- TPU precision: default vs `highest` drift: vertical velocity 9.4e-2, precip 7.2e-2, 10m u 1.9e-2,
  q 1.3e-2, MSLP 3.6e-3, 2m T 3.0e-3, T 1.9e-3, geopotential 7e-5. That is ~8x the A10 TF32 drift and
  ~3x the A10 bf16-attention drift -> TPU default is effectively bf16-input matmuls. The 59.5 ms is
  NOT precision-matched to the A10's 132 ms.
- Reinterpretation (review): 1024/1024/512 is exactly the shipped Mini forward config -> 61.0 ms. So
  "191 -> 59.5 by tuning" is really "the notebook's fit-on-v5e override is 3x slower for inference than
  Google's own config; our sweep adds +2.6%". Internally precision-matched (all default), but the
  sweep never compared outputs across block sizes (splash masks elementwise in partial blocks, so it
  should match up to summation order - verify).
- `data/tpu_parity_{default,highest}_prediction.nc` are downloaded (git-ignored) but NOT yet compared
  against the A10 JAX fp32 reference (script 06; regenerate the reference on a GPU box first). Expect
  TPU-highest vs A10-fp32 ~1e-5-ish if both platforms agree.

## 8b. Full 0.25-deg model on the A10 (24 GB): estimate, scripts ready, NOT yet run
- Geometry (counted with the official mesh code; HF config agrees: 1,618,841 g2m edges): grid
  1,038,240 points, mesh 40,962, grid->mesh 1.62M edges, mesh->grid 3,114,720 (3/point). Mini:
  65,160 / 10,242 / 101,892 / 195,480.
- B0: cannot fit (30 GB of scores per layer alone).
- Optimized attention: fits, but the next wall is the mesh->grid decoder: the port computes a
  768-wide fp32 message per edge in one go = 9.6 GB/tensor, 2-3 alive -> est. 25-35 GB total ->
  likely OOM. Consistency check: same arithmetic on Mini (~1.2 GB edge tensors + weights) ~ the
  measured 1.83 GB peak, i.e. the decoder probably already sets the Mini peak (explains the small
  -21%). Port config has `chunk_size`, but our code path never uses it.
- Fix: `scripts/chunked_graph.py` (`install_chunked(model, chunk)`; script 05 `--chunk N`): row-chunks
  every grid/edge-scale layer (ConditionedMlps; g2m by mesh-node ranges with all their incoming
  edges; m2g by grid-point ranges; output head). All row-wise -> exact up to rounding. Est. 10-14 GB.
  Verified on CPU: `scripts/test_chunked_graph.py` (random model, uneven fan-in: float64 diff 0,
  fp32 8.5e-8; a deliberately broken variant fails it); real Mini checkpoint + input, chunk 8192 vs
  off: diff 0 in fp32 on CPU (a mutation check confirmed the patched path runs).
- `mesh_flex_attention.py`: mask reorder frees the unordered copy first (1.7 GB per N x N copy at
  0.25 deg). Script 05 drops the port's blocked mask (1.27 GB) after the first call when `--mesh-attn`
  is set, and reports `warmup_peak_gb` (incl. setup) besides steady-state `peak_gb`.
- Script 05: `--model full` (kashif/weathernext-cyclones rev `<2024` subfolder `model1` =
  WeatherNextCyclones_<2024_model1), auto-downloads the 0.25-deg input (1.27 GB); `--memory-snapshot`
  (view at pytorch.org/memory_viz); OOM is caught and saved as results JSON with status "oom".
  Script 03: `--model full`, `--n-timed`, OOM saved likewise (XLA error text incl. requested size).
- Runner: `bash scripts/run_full_model.sh` (steps 0-4: unit test; Mini chunked parity + overhead; B0
  full [expect OOM]; full unchunked [expect decoder OOM]; full chunked [expect fit]); `... all` adds
  compile and an fp32 parity vs official JAX on CPU (no GPU reference exists; needs ~100+ GB host RAM).
- Not checked offline: Hilbert `node_order` end-to-end on the 40,962-node mesh, and whether the TUNED
  flex tiles are still best at 6 heads / 32 hops (re-run script 12 on the full mesh).

## 9. Open items (priority order)

0. Full model on A10: `bash scripts/run_full_model.sh` (sec. 8b).
0b. TPU, ~2 min: time the shipped config unmodified (no notebook override) as THE TPU baseline; in the
   same session compare outputs of 128-block vs 512/1024/512 (expect ~1e-6, not ~1e-2).
0c. A10: parity for B0 at default precision and for the exact 132 ms config (TF32 drift of the timed
   configs; both expected ~1e-2).
1. TPU precision-matched timing (use shipped blocks, not 128): `16_tpu_tuning_parity.py --skip-parity --sweep-precision high|highest
   --blocks 128,128,128 512,1024,512` (`high` = 3-pass bf16, closest to TF32). Then the fair table is
   A10 TF32 (132) vs TPU `high`, plus each platform's fastest with its measured drift.
2. TPU rollout with tuned blocks (script 15 hard-codes 128 blocks; parametrize) to replace the 3.6 s
   extrapolation.
3. A10 bf16 + meshflex Hilbert + compile, with a bf16 script-12 sweep (halves regs/smem; targets the
   register-spill limiter in sec. 5.12). Compare against TPU default at similar drift.
4. Multi-step accuracy: rollout skill vs truth for exact vs bf16 (and TPU default) - needed before any
   claim that lower precision is free. Sample data only has 30-step truth.
5. Full 0.25 deg model (40,962 nodes, 32-hop, d768, 24 layers) on H100: re-run scripts 10/12 (tiles,
   ordering, tuning are shape/GPU specific); `node_order` supports splits 6.
6. Optional: JAX triblockdiag with concatenated blocks (does XLA then fuse softmax?) for a fairer
   JAX-vs-PyTorch statement.
7. Milton visuals (script 14): tracker is grid-argmin MSLP (staircase tracks); sub-grid interpolation
   would smooth. Intensity is far too weak (975-995 vs observed 895-910 hPa): 1 deg resolution limit.

## 10. File map

Scripts: 01 mask/tile waste analysis (CPU) | 02 tile-map plot | 03 B0 JAX timing/profile/parity ref |
04 JAX trace + HLO -> stage breakdown | 05 B1 torch: all variants (`--attn`, `--compile`, `--tf32`,
`--attn-dtype`, `--mesh-attn`, `--flex-tuned`, `--nvtx-probe`, `--nvtx-step`, `--profile`, `--noise`,
`--n-timed`) | 06 parity vs JAX nc | 07 ncu CSV -> bytes/time by kernel family | 08 inspect port
BlockMask + flex launch grid | 09 one-layer Flex microbench (port layout) | 10 node orderings (CPU) |
11 one-layer full-mesh Flex per ordering | 12 Flex kernel_options sweep | 13 ensemble rollout + tracker
+ fields | 14 render Milton plots/GIF | 15 TPU official timing + rollout | 16 TPU parity + splash sweep |
`mesh_flex_attention.py` drop-in processor replacement | `chunked_graph.py` (+ `test_chunked_graph.py`)
row-chunked grid layers | `run_full_model.sh` full-model runner | `attn_precision.py` bf16 attention patch |
`run_sweep.sh`, `run_meshflex.sh`, `run_bytes_probe.sh` batch runners | `setup_instance.sh`.

Results: `b0_*` (B0 json, breakdown, ncu CSV) | `b1_torch_mini_NVIDIA_A10_<variant>_<prec>_<tag>.json`
(all torch runs) | `flex_microbench.md`, `flex_inspect_*.json`, `flex_tuning_fp32.csv` |
`ordering_splits5_k16_t128.json` | `rollout_a10_m32_b8.json` | `tpu_b0_results.json`,
`tpu_tuning_parity.json` | `milton_*.png`, `forecast_mslp_wind.gif` | `profiles/` (git-ignored:
traces, `bytes_*.csv`, `.ncu-rep[z]` incl. `ncu_b1_attn_probe` and `ncu_tuned_attn_probe` for
before/after comparison in the Nsight GUI).
