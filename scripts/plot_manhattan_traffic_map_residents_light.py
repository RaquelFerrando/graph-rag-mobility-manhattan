"""
Light-background variant of plot_manhattan_traffic_map_residents.py, for the README
(the dark version reads well on slides but poorly against a white page). Same data,
same panels, just a light basemap/colormap/text palette instead of a dark one.
"""
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from xyzservices import TileProvider

LIGHT_BASEMAP = TileProvider(
    name="Esri.WorldLightGrayBase",
    url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution="Esri",
)

BG = "#FFFFFF"
TEXT = "#1A1A1A"
STREET_DIM = "#C7CAD1"
CMAP = LinearSegmentedColormap.from_list("traffic_light", ["#D8D8D8", "#E8A33D", "#D9541E", "#8B1E1E"])
MARGIN_FRAC = 0.03

real_full = gpd.read_file("data/geo/replica-manhattan-roads-real.geojson").to_crs(3857)
real_res = gpd.read_file("results/manhattan/exp2_traffic_residents_edges.geojson").to_crs(3857)
sim = gpd.read_file("results/manhattan/exp2_traffic_edges.geojson").to_crs(3857)

x0, y0, x1, y1 = real_full.total_bounds
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
    d = gdf[gdf.geometry.notna()].copy()
    d = d[d.geometry.geom_type == "LineString"]
    sqrt_v = np.sqrt(d[col].astype(float))
    vmin, vmax = sqrt_v.min(), sqrt_v.max()
    d["_norm"] = (sqrt_v - vmin) / (vmax - vmin)
    d = d.sort_values("_norm")
    segs = [list(g.coords) for g in d.geometry]
    lw = 0.7 + d["_norm"].values * 7.5
    alpha = 0.45 + d["_norm"].values * 0.55
    return segs, lw, alpha, d["_norm"].values


def draw_dim_backdrop(ax):
    segs_all = [list(g.coords) for g in real_full.geometry if g is not None and g.geom_type == "LineString"]
    ax.add_collection(LineCollection(segs_all, linewidths=0.4, colors=STREET_DIM))


def add_basemap(ax):
    ctx.add_basemap(ax, source=LIGHT_BASEMAP, zoom=13, attribution=False)


# --- Panel A: real RESIDENT-ONLY traffic (routed by us) ---
ax = axes[0]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
add_basemap(ax)
draw_dim_backdrop(ax)
segs, lw, alpha, norm_vals = to_segments_sorted(real_res, "real_resident_count")
colors = [(*CMAP(v)[:3], a) for v, a in zip(norm_vals, alpha)]
ax.add_collection(LineCollection(segs, linewidths=lw, colors=colors))
ax.set_title("Real Manhattan RESIDENTS only (3,550 real trips, routed by us)", color=TEXT, fontsize=15, pad=14)

# --- Panel B: simulated traffic ---
ax = axes[1]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
add_basemap(ax)
draw_dim_backdrop(ax)
segs_sim, lw_s, alpha_s, norm_vals_s = to_segments_sorted(sim, "sim_count")
colors_s = [(*CMAP(v)[:3], a) for v, a in zip(norm_vals_s, alpha_s)]
ax.add_collection(LineCollection(segs_sim, linewidths=lw_s, colors=colors_s))
ax.set_title("Our simulation (1,000 residents)", color=TEXT, fontsize=15, pad=14)

fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig("results/manhattan/exp2_traffic_map_residents_light.png", dpi=180, facecolor=BG, bbox_inches="tight")
print("saved results/manhattan/exp2_traffic_map_residents_light.png")
