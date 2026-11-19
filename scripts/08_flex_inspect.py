"""Step 8: what is FlexAttention actually doing in this model?

1. Captures the BlockMask the port hands to flex_attention and reports how sparse
   it is: of all (query-tile, key-tile) pairs, how many are skipped, how many are
   computed with the mask function, and how many are fully unmasked.
2. Profiles one step and reads the launch grid of the flex kernel, which tells
   us the query-tile size (BLOCK_M) Triton picked.

    python scripts/08_flex_inspect.py [--attn-dtype fp32|bf16]
"""
import argparse
import importlib.util
import json
import math
import pathlib

import torch
from transformers.integrations import flex_attention as hf_flex
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Reuse the setup code from script 05 (its file name starts with a digit, so it
# can't be imported with a normal import statement).
spec = importlib.util.spec_from_file_location("b1", pathlib.Path(__file__).parent / "05_b1_torch.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

captured = {}


def capture(original):
    def wrapper(module, query, key, value, attention_mask, **kwargs):
        if "mask" not in captured:
            captured["mask"] = attention_mask
            captured["q_shape"] = tuple(query.shape)
            captured["kv_shape"] = tuple(key.shape)
            captured["dtype"] = str(query.dtype)
        return original(module, query, key, value, attention_mask, **kwargs)
    return wrapper


def main(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.attn_dtype != "fp32":
        from attn_precision import patch_attention_dtype
        patch_attention_dtype({"bf16": torch.bfloat16, "fp16": torch.float16}[a.attn_dtype])
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = capture(ALL_ATTENTION_FUNCTIONS["flex_attention"])

    dev = torch.device("cuda")
    processor = b1.WeatherNext2FeatureExtractor.from_pretrained(b1.REPO)
    model = b1.WeatherNext2ForWeatherForecasting.from_pretrained(
        b1.REPO, attn_implementation="flex_attention", dtype=torch.float32).to(dev).eval()
    import numpy as np, xarray
    ds = xarray.load_dataset(b1.DATA)
    state = b1.load_state(processor, ds)
    t = torch.as_tensor(ds["seconds_since_epoch"].isel(time=2).values, dtype=torch.int64)
    inputs = processor(state, seconds_since_epoch=t, noise=torch.from_numpy(np.load("data/parity_noise.npy")),
                       device=dev)
    with torch.inference_mode():
        model(**inputs)          # warmup + capture
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(**inputs)
            torch.cuda.synchronize()

    m = captured["mask"]
    print(f"attention inputs: q {captured['q_shape']}  kv {captured['kv_shape']}  {captured['dtype']}")
    print(f"mask object: {type(m).__name__}")
    report = {"q_shape": captured["q_shape"], "kv_shape": captured["kv_shape"], "dtype": captured["dtype"]}
    if hasattr(m, "kv_num_blocks"):
        bq, bkv = m.BLOCK_SIZE
        nq = math.ceil(captured["q_shape"][-2] / bq)
        nkv = math.ceil(captured["kv_shape"][-2] / bkv)
        partial = m.kv_num_blocks.float()                 # tiles computed WITH mask_mod
        full = (m.full_kv_num_blocks.float() if m.full_kv_num_blocks is not None
                else torch.zeros_like(partial))           # tiles computed without mask_mod
        total_pairs = nq * nkv
        print(f"block size {bq}x{bkv}; per (batch, head): {nq} query tiles x {nkv} key tiles = {total_pairs} pairs")
        print(f"  mask tensor shape {tuple(m.kv_num_blocks.shape)}")
        print(f"  computed with mask_mod (partial): {partial.mean().item() * nq:8.0f} pairs ({partial.mean().item() / nkv:.1%})")
        print(f"  computed without mask   (full):   {full.mean().item() * nq:8.0f} pairs ({full.mean().item() / nkv:.1%})")
        skipped = 1 - (partial.mean().item() + full.mean().item()) / nkv
        print(f"  skipped:                          {skipped:.1%}")
        report.update(block=[bq, bkv], partial_frac=partial.mean().item() / nkv,
                      full_frac=full.mean().item() / nkv, skipped_frac=skipped)

    # Flex kernel launch grid -> query tile size actually used by the kernel.
    for e in prof.events():
        if "flex_attention" in e.name and e.device_type.name == "CUDA":
            print(f"kernel: {e.name[:70]}  {e.device_time / 1e3:.2f} ms")
            break
    trace = pathlib.Path("logs/flex_trace.json")
    prof.export_chrome_trace(str(trace))
    for ev in json.load(open(trace))["traceEvents"]:
        if ev.get("cat") == "kernel" and "flex_attention" in ev.get("name", ""):
            grid, block = ev["args"].get("grid"), ev["args"].get("block")
            bm = math.ceil(captured["q_shape"][-2] / grid[0]) if grid else None
            print(f"launch grid {grid} x block {block}  -> ~{bm} queries per program (BLOCK_M)")
            report.update(grid=grid, threads=block, block_m=bm)
            break
    out = pathlib.Path(f"results/flex_inspect_{a.attn_dtype}.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    main(ap.parse_args())
