"""Step 2: see the real Mini attention mask the way a kernel sees it.

Reads the .npz saved by 01_mask_analysis.py and draws two panels:
  left : the mask at TILE resolution (one pixel = one 128x128 tile).
         Color = how full the tile is (fraction of its cells that are needed).
         Dashed boxes = the blocks the GPU triblockdiag path computes.
  right: histogram of fill fraction over the tiles a splash-style kernel would
         compute. Lots of mass near 0 = lots of mostly-empty tiles = waste.

Run:
    python scripts/01_mask_analysis.py --splits 5 --khop 16
    python scripts/02_plot_tile_map.py results/tiles_splits5_k16_t128.npz
"""
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SURFACE, INK, INK2, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
# Sequential: one hue, light -> dark. Empty tiles stay the surface color.
CMAP = LinearSegmentedColormap.from_list("fill", ["#cfe1f7", BLUE, "#0d3a73"])
CMAP.set_bad(SURFACE)

path = pathlib.Path(sys.argv[1])
d = np.load(path)
counts, n, k, T = d["tile_counts"], int(d["n"]), int(d["khop"]), int(d["tile"])
bw, useful = int(d["bandwidth"]), int(d["useful"])

nt = counts.shape[0]
# Cells per tile: edge tiles are smaller because N isn't a multiple of T.
sizes = np.minimum(T, n - np.arange(nt) * T)
fill = counts / np.outer(sizes, sizes)
touched = counts > 0

# Waste factors (same definitions as script 01).
splash = touched.sum() * T * T / useful
n_blocks = -(-n // bw)
tri = n_blocks * 3 * bw * bw / useful

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": "#d9d8d3"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6),
                               gridspec_kw={"width_ratios": [1.1, 1]},
                               facecolor=SURFACE)

# Left: tile map.
ax1.set_facecolor(SURFACE)
im = ax1.imshow(np.where(touched, fill, np.nan), cmap=CMAP, vmin=0, vmax=1,
                interpolation="nearest", extent=(0, n, n, 0))
for b in range(n_blocks):                         # triblockdiag blocks
    for nb in (b - 1, b, b + 1):
        if 0 <= nb < n_blocks:
            ax1.add_patch(Rectangle((nb * bw, b * bw), bw, bw, fill=False,
                                    ls=(0, (4, 3)), lw=1, ec=INK2))
ax1.set_xlim(0, n); ax1.set_ylim(n, 0)
ax1.set_xlabel("key node j (RCM order)"); ax1.set_ylabel("query node i (RCM order)")
ax1.set_title(f"{T}x{T} tiles, colored by fill. Dashed = GPU triblockdiag blocks",
              fontsize=11, color=INK2, loc="left")
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
cb.set_label("fraction of tile that is needed"); cb.outline.set_visible(False)

# Right: how full are the tiles we compute?
ax2.set_facecolor(SURFACE)
f = fill[touched]
ax2.hist(f, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor=SURFACE, lw=2)
med = np.median(f)
ax2.axvline(med, color=INK2, ls=(0, (4, 3)), lw=1)
ax2.text(0.98, 0.95, f"dashed line: median tile is {med:.0%} full", color=INK2,
         transform=ax2.transAxes, ha="right", va="top")
ax2.set_xlabel("fraction of tile that is needed")
ax2.set_ylabel("number of computed tiles")
ax2.set_title("How full are the tiles a splash-style kernel computes?",
              fontsize=11, color=INK2, loc="left")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.grid(axis="y", color="#ebeae5", lw=0.8); ax2.set_axisbelow(True)

fig.suptitle(f"WeatherNext 2 attention mask: {n:,} mesh nodes, {k}-hop window.  "
             f"Waste: triblockdiag {tri:.1f}x, {T}-tiles {splash:.1f}x, "
             f"dense {n * n / useful:.1f}x", x=0.02, ha="left", fontsize=13)
fig.tight_layout()
out = path.with_suffix(".png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"saved -> {out}")
