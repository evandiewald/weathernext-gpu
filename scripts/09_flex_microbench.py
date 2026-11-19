"""Step 9: isolate FlexAttention from the rest of the model.

Real shapes and the real mesh mask from one transformer layer of the Mini model
(4 node-blocks x 4 heads, 2577 queries, 7731 keys = own block + 2 neighbours,
head dim 128), random Q/K/V. Variants:

  A  port            : block-sparse tiles + mask_mod that looks up the dense mask
  B  sparse, no mod  : same skipped tiles, but no per-element mask lookup
                       (WRONG numerics on purpose: shows the cost of the lookup)
  C  dense, no mod   : every tile computed, no mask (shows the value of skipping)
  D  SDPA, dense mask: PyTorch's fused attention with a boolean mask
  F  arithmetic mod  : same tiles, mask_mod with no memory reads (WRONG on purpose):
                       separates "masking at all" from "reading the mask"

    python scripts/09_flex_microbench.py --dtype fp32   # TF32 on
    python scripts/09_flex_microbench.py --dtype bf16
"""
import argparse
import importlib.util
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import xarray
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

spec = importlib.util.spec_from_file_location("b1", pathlib.Path(__file__).parent / "05_b1_torch.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

flex = torch.compile(flex_attention)   # flex must be compiled to run its Triton kernel


def bench(fn, n=20) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def main(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]

    # The real geometry mask: [num_blocks, 1, block, 3 * block] booleans.
    processor = b1.WeatherNext2FeatureExtractor.from_pretrained(b1.REPO)
    ds = xarray.load_dataset(b1.DATA)
    t = torch.as_tensor(ds["seconds_since_epoch"].isel(time=2).values, dtype=torch.int64)
    inputs = processor(b1.load_state(processor, ds), seconds_since_epoch=t, device=dev)
    mask = inputs["attention_mask"]
    nb, _, lq, lkv = mask.shape
    H, D = 4, 128
    print(f"mask {tuple(mask.shape)}, density {mask.float().mean().item():.1%} "
          f"(fraction of the {lq}x{lkv} score entries that are needed)")

    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(nb, H, lq, D, device=dev, dtype=dtype, generator=g)
    k = torch.randn(nb, H, lkv, D, device=dev, dtype=dtype, generator=g)
    v = torch.randn(nb, H, lkv, D, device=dev, dtype=dtype, generator=g)

    # Same semantics as the port's banded_mask_function: look up the dense mask.
    # (The port's version, called standalone, trips a torch.compile restriction.)
    def port_mod(b, h, q_idx, kv_idx):
        return mask[b % nb, 0, q_idx, kv_idx]
    bm_port = create_block_mask(port_mod, B=nb, H=None, Q_LEN=lq, KV_LEN=lkv, device=dev)
    bm_nomod = BlockMask.from_kv_blocks(bm_port.kv_num_blocks, bm_port.kv_indices,
                                        BLOCK_SIZE=bm_port.BLOCK_SIZE, mask_mod=None,
                                        seq_lengths=(lq, lkv))
    sdpa_mask = mask.expand(nb, H, lq, lkv)

    # F: same tiles, a mask_mod that only does arithmetic (no memory reads).
    # Splits the cost of A-vs-B into "masking at all" vs "reading the mask".
    def arith_mod(b, h, q_idx, kv_idx):
        return (q_idx * 7 + kv_idx * 13) % 10 != 0
    bm_arith = BlockMask.from_kv_blocks(bm_port.kv_num_blocks, bm_port.kv_indices,
                                        BLOCK_SIZE=bm_port.BLOCK_SIZE, mask_mod=arith_mod,
                                        seq_lengths=(lq, lkv))

    useful_flops = 4 * mask.float().sum().item() * H * D          # QK^T + PV, needed entries
    rows = [
        ("A port: sparse tiles + dense-mask lookup", lambda: flex(q, k, v, block_mask=bm_port)),
        ("B sparse tiles, no lookup (wrong on purpose)", lambda: flex(q, k, v, block_mask=bm_nomod)),
        ("C all tiles, no mask (wrong on purpose)", lambda: flex(q, k, v)),
        ("F sparse tiles, arithmetic mask (wrong on purpose)", lambda: flex(q, k, v, block_mask=bm_arith)),
        ("D SDPA with boolean mask", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)),
    ]
    print(f"\n{a.dtype} (TF32 matmuls on), one layer of attention:")
    for name, fn in rows:
        with torch.inference_mode():
            ms = bench(fn)
        print(f"  {name:48s} {ms:7.2f} ms   useful TFLOP/s {useful_flops / ms / 1e9:5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    main(ap.parse_args())
