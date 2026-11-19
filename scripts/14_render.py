"""Step 14: visuals from the 15-day ensemble (data/rollout_m32.npz from script 13).

  results/milton_tracks.png   32 ensemble tracks of Hurricane Milton (MSLP-minimum tracker),
                              first 5 days, with the observed best track (NOAA IBTrACS)
  results/milton_pressure.png storm-center pressure vs time, ensemble vs observed
  results/forecast_mslp_wind.gif  member 0, 15 days: 10 m wind speed + MSLP contours

    curl -L -o data/ibtracs_last3years.csv https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv
    python scripts/14_render.py
"""
import csv
import datetime as dt
import pathlib

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, OBS = "#2a78d6", "#0b0b0b"
DAYS_TRACKED = 5
d = np.load("data/rollout_m32.npz")
tracks, lats, lons = d["tracks"], d["lats"], d["lons"]
init = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(d["init_seconds"]))
steps = tracks.shape[1]
times = [init + dt.timedelta(hours=6 * (s + 1)) for s in range(steps)]
n_track = DAYS_TRACKED * 4
lon_w = lambda x: (x + 180) % 360 - 180                     # 0..360 -> -180..180


def observed_milton():
    """6-hourly best-track points of Milton (2024) from IBTrACS."""
    pts = []
    with open("data/ibtracs_last3years.csv") as f:
        rows = csv.reader(f)
        header = next(rows); next(rows)                      # 2nd row = units
        i = {k: header.index(k) for k in ("NAME", "SEASON", "ISO_TIME", "LAT", "LON", "WMO_PRES", "USA_PRES")}
        for r in rows:
            if r[i["NAME"]] == "MILTON" and r[i["SEASON"]] == "2024":
                t = dt.datetime.strptime(r[i["ISO_TIME"]], "%Y-%m-%d %H:%M:%S")
                if t.minute == 0 and t.hour % 6 == 0 and t >= init:
                    p = r[i["USA_PRES"]].strip() or r[i["WMO_PRES"]].strip()
                    pts.append((t, float(r[i["LAT"]]), float(r[i["LON"]]), float(p) if p else np.nan))
    return pts


obs = observed_milton()
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})

# 1. Track spaghetti
fig = plt.figure(figsize=(10, 7), facecolor=SURFACE)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-100, -70, 15, 37], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#efeee9")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK2)
ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor=GRID)
for m in range(tracks.shape[0]):
    ax.plot(lon_w(tracks[m, :n_track, 1]), tracks[m, :n_track, 0], color=BLUE, alpha=0.35, lw=1,
            transform=ccrs.PlateCarree())
mean_lat = tracks[:, :n_track, 0].mean(0)
mean_lon = lon_w(np.rad2deg(np.arctan2(np.sin(np.deg2rad(tracks[:, :n_track, 1])).mean(0),
                                         np.cos(np.deg2rad(tracks[:, :n_track, 1])).mean(0))))
ax.plot(mean_lon, mean_lat, color=BLUE, lw=3, label="ensemble mean (32 members)", transform=ccrs.PlateCarree())
ax.plot(mean_lon[3::4], mean_lat[3::4], "o", color=BLUE, ms=6, transform=ccrs.PlateCarree())
o = [p for p in obs if p[0] <= times[n_track - 1]]
ax.plot([p[2] for p in o], [p[1] for p in o], color=OBS, lw=2.5, ls=(0, (5, 3)),
        label="observed (NOAA IBTrACS)", transform=ccrs.PlateCarree())
ax.plot([p[2] for p in o if p[0].hour == 0], [p[1] for p in o if p[0].hour == 0], "s", color=OBS, ms=5,
        transform=ccrs.PlateCarree())
ax.legend(loc="lower right", frameon=False)
ax.set_title(f"Hurricane Milton: WeatherNext 2 Mini, 32-member ensemble\n"
             f"init {init:%Y-%m-%d %HZ}, first {DAYS_TRACKED} days; markers every 24 h",
             loc="left", fontsize=12)
fig.savefig("results/milton_tracks.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# 2. Central pressure
fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
hours = np.arange(1, n_track + 1) * 6
for m in range(tracks.shape[0]):
    ax.plot(hours, tracks[m, :n_track, 2], color=BLUE, alpha=0.3, lw=1)
ax.plot(hours, np.median(tracks[:, :n_track, 2], 0), color=BLUE, lw=3, label="ensemble median")
oh = [(p[0] - init).total_seconds() / 3600 for p in o]
ax.plot(oh, [p[3] for p in o], color=OBS, lw=2.5, ls=(0, (5, 3)), label="observed")
ax.set_xlabel("forecast hour"); ax.set_ylabel("minimum sea-level pressure (hPa)")
ax.set_xticks(range(0, n_track * 6 + 1, 24))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#ebeae5", lw=0.8)
ax.legend(frameon=False)
ax.set_title("Milton central pressure. A 1-degree model can't resolve the 895 hPa core",
             loc="left", fontsize=12)
fig.savefig("results/milton_pressure.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# 3. Animation: member 0, North Atlantic / Americas
frames = []
lon2d, lat2d = np.meshgrid(lons, lats)
wind, mslp = d["wind10"][:, 1].astype(np.float32), d["mslp"][:, 1].astype(np.float32)
for s in range(steps):
    fig = plt.figure(figsize=(10, 6), facecolor=SURFACE)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-130, -10, 5, 60], crs=ccrs.PlateCarree())
    pc = ax.pcolormesh(lons, lats, wind[s], cmap="Blues", vmin=0, vmax=30, transform=ccrs.PlateCarree())
    ax.contour(lon2d, lat2d, mslp[s], levels=np.arange(940, 1050, 4), colors=INK2, linewidths=0.5,
               transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor=INK)
    cb = fig.colorbar(pc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("10 m wind speed (m/s)"); cb.outline.set_visible(False)
    ax.set_title(f"WeatherNext 2 Mini (member 0): 10 m wind + sea-level pressure (4 hPa contours)\n"
                 f"valid {times[s]:%Y-%m-%d %HZ}  (+{6 * (s + 1)} h)", loc="left", fontsize=11)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
imageio.mimsave("results/forecast_mslp_wind.gif", frames, duration=0.15, loop=0)
print("saved results/milton_tracks.png, results/milton_pressure.png, results/forecast_mslp_wind.gif")
