"""
Side-by-side street map: real Manhattan traffic (trip_count, all 48,153 real
segments, includes non-residents) vs. our simulated traffic (sim_count, only
the 22,865 segments any of the 1,000 residents-only agents actually routed
through). Kept separately from plot_manhattan_traffic_map_residents.py (the
fair residents-vs-residents comparison) since this "vs everyone" version is
still referenced on its own (the original, less-fair comparison, used to show
the before/after of the methodology fix).

v2: same visual upgrade as the residents version - real dark basemap (Esri
World Dark Gray Canvas, no API key needed), tighter zoom, thicker/brighter
lines - requested after the residents map got this treatment and the
"everyone" map was still on the old flat-black style.
"""
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from xyzservices import TileProvider

DARK_BASEMAP = TileProvider(
    name="Esri.WorldDarkGrayBase",
    url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution="Esri",
)

BG = "#0B0E14"
STREET_DIM = "#3A3E48"
CMAP = LinearSegmentedColormap.from_list("traffic", ["#3A3D42", "#B8862E", "#FFC93C", "#FFF6D9"])
MARGIN_FRAC = 0.03

real = gpd.read_file("data/geo/replica-manhattan-roads-real.geojson").to_crs(3857)
sim = gpd.read_file("results/manhattan/exp2_traffic_edges.geojson").to_crs(3857)

x0, y0, x1, y1 = real.total_bounds
dx, dy = (x1 - x0) * MARGIN_FRAC, (y1 - y0) * MARGIN_FRAC
xlim = (x0 - dx, x1 + dx)
ylim = (y0 - dy, y1 + dy)

fig, axes = plt.subplots(1, 2, figsize=(20, 13), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def to_segments_sorted(gdf, col):
    """Same trick as the paper's visualize_trip_data: sqrt-normalize, then sort
    ascending so the busiest streets draw LAST (on top) - that's most of why
    its version reads so much better than a flat log-alpha blend."""
    d = gdf[gdf.geometry.notna()].copy()
    d = d[d.geometry.geom_type == "LineString"]
    sqrt_v = np.sqrt(d[col].astype(float))
    vmin, vmax = sqrt_v.min(), sqrt_v.max()
    d["_norm"] = (sqrt_v - vmin) / (vmax - vmin)
    d = d.sort_values("_norm")  # low first, high last (drawn on top)
    segs = [list(g.coords) for g in d.geometry]
    lw = 0.7 + d["_norm"].values * 7.5
    alpha = 0.45 + d["_norm"].values * 0.55
    return segs, lw, alpha, d["_norm"].values


def draw_dim_backdrop(ax):
    segs_all = [list(g.coords) for g in real.geometry if g is not None and g.geom_type == "LineString"]
    ax.add_collection(LineCollection(segs_all, linewidths=0.4, colors=STREET_DIM))


def add_basemap(ax):
    ctx.add_basemap(ax, source=DARK_BASEMAP, zoom=13, attribution=False)


# --- Panel A: real traffic, every segment, sqrt-normalized width/color/alpha ---
ax = axes[0]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
add_basemap(ax)
segs, lw, alpha, norm_vals = to_segments_sorted(real, "trip_count")
colors = [(*CMAP(v)[:3], a) for v, a in zip(norm_vals, alpha)]
ax.add_collection(LineCollection(segs, linewidths=lw, colors=colors))
ax.set_title("Real Manhattan traffic (all residents + everyone else)", color="#F2F0EA", fontsize=15, pad=14)

# --- Panel B: simulated traffic - full network dim, sim edges highlighted ---
ax = axes[1]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
add_basemap(ax)
draw_dim_backdrop(ax)
segs_sim, lw_s, alpha_s, norm_vals_s = to_segments_sorted(sim, "sim_count")
colors_s = [(*CMAP(v)[:3], a) for v, a in zip(norm_vals_s, alpha_s)]
ax.add_collection(LineCollection(segs_sim, linewidths=lw_s, colors=colors_s))
ax.set_title("Our simulation (1,000 residents only)", color="#F2F0EA", fontsize=15, pad=14)

fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig("results/manhattan/exp2_traffic_map_vs_everyone.png", dpi=180, facecolor=BG, bbox_inches="tight")
print("saved results/manhattan/exp2_traffic_map_vs_everyone.png")
