# FlexAttention microbenchmark (scripts/09_flex_microbench.py), A10, one layer of Mini attention
Shapes: 4 node-blocks x 4 heads, 2577 queries, 7731 keys (own block + 2 neighbours), head dim 128.
Mask density 10.3% of entries. BlockMask (128x128 tiles): 45.1% of tiles skipped, 54.9% computed, 0% "full".

| variant | fp32 (TF32) ms | bf16 ms | output |
|---|---|---|---|
| A port: sparse tiles + dense bool mask lookup in mask_mod | 17.4 | 10.2 | correct (vs SDPA: 1.2e-6) |
| B same tiles, no mask_mod | 4.8 | 1.2 | wrong on purpose |
| F same tiles, arithmetic-only mask_mod (no memory reads) | 5.4 | 1.6 | wrong on purpose |
| C all tiles, no mask | 7.8 | 2.3 | wrong on purpose |
| D SDPA, boolean mask | 27.1 | 12.0 | reference |

Bit-packing the mask (8x fewer bytes, row-major int32 words) only went 22.6 -> 18.4 ms (fp32) /
13.1 -> 10.5 ms (bf16), and was wrong in one tile (block 0, queries 0-127) under flex although it
unpacks correctly on the host. Dropped.

Reading: skipping tiles is worth ~1.6-1.9x (B vs C). Masking per element costs little (F vs B).
Reading the mask from memory per element costs 3-6x (A vs F) - and it's not the byte count
(packing barely helped): it's ~16K scattered loads per computed tile.
