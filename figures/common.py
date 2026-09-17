"""Plot style, network geometry and the spatial error renderer shared by the figure scripts."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED = REPO / "outputs/predictions"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#DDDDDD",
    "lines.linewidth": 1.4,
})

C_GT = "#1A1A2E"
C_MODEL = "#1B6CA8"
C_SENSOR_SPATIAL = "#7B2FBE"
_ERR_CMAP = LinearSegmentedColormap.from_list(
    "err", ["#2166AC", "#92C5DE", "#FDDBC7", "#EF6548", "#B2182B"])

_TUI = "data/raw/tuindorp/Tuindorp development - 1 min resolution/networks/"
TOPO = {
    "bellinge": str(REPO / "data/interim/bellinge/topology/BellingeSWMM_node.csv"),
    "heusden": str(REPO / "data/raw/heusden/Detailed_heusden_topology/1D rioleringsmodel Heusden!_node.csv"),
    "tuindorp": str(REPO / _TUI / "Tuindorp development - 1 min resolution_node.csv"),
}
MH_TYPE = {"bellinge": "Manhole", "heusden": "Manhole", "tuindorp": "Junction"}
COND_DIR = {
    "bellinge": str(REPO / "data/interim/bellinge/topology"),
    "heusden": str(REPO / "data/raw/heusden/Detailed_heusden_topology"),
    "tuindorp": str(REPO / _TUI),
}


def load_coords(net):
    df = pd.read_csv(TOPO[net], encoding="latin-1")
    type_col = next(c for c in df.columns if c.lower() in ("node_type", "type"))
    mh = df[df[type_col] == MH_TYPE[net]].reset_index(drop=True)
    return mh["x"].values.astype(float), mh["y"].values.astype(float)


def _save(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight", dpi=200)
    print(f"saved {FIG_DIR / name}.pdf")
    plt.close(fig)


def load_all_edges(net):
    """Load edges from ALL hydraulic link types (conduit, pump, weir, channel, etc.)
    using ALL node types (Manhole, Storage, Outfall) for coordinate lookup."""
    try:
        nodes = pd.read_csv(TOPO[net], encoding="latin-1")
        id2pos = {str(r["node_id"]): (float(r["x"]), float(r["y"]))
                  for _, r in nodes.iterrows()
                  if pd.notna(r.get("x")) and pd.notna(r.get("y"))}
        edges = []
        link_files = glob.glob(os.path.join(COND_DIR[net], "*.csv"))
        link_files = [f for f in link_files if "_node" not in f]
        for lf in link_files:
            try:
                df = pd.read_csv(lf, encoding="latin-1")
            except Exception:
                continue
            if df.empty:
                continue
            us_col = next((c for c in df.columns if "us_node" in c.lower()
                           or c.lower() in ("upstream_node", "from_node", "us")), None)
            ds_col = next((c for c in df.columns if "ds_node" in c.lower()
                           or c.lower() in ("downstream_node", "to_node", "ds")), None)
            if us_col is None or ds_col is None:
                continue
            for _, row in df.iterrows():
                u, v = str(row[us_col]), str(row[ds_col])
                if u in id2pos and v in id2pos:
                    edges.append((id2pos[u], id2pos[v]))
        # deduplicate
        return list(dict.fromkeys(edges))
    except Exception as e:
        print(f"  load_all_edges failed for {net}: {e}")
        return []


def _draw_spatial_error_ax(ax, key):
    """Populate a single axes with the spatial error map for network `key`."""
    d    = np.load(PRED / f"{key}.npz", allow_pickle=True)
    mae  = d["mae_node_cm"]
    mask = d["non_sensor"]

    cx, cy = load_coords(key)
    n = len(mae)
    cx, cy = cx[:n], cy[:n]

    # Network skeleton
    lw    = 0.40 if key == "tuindorp" else 0.70
    edges = load_all_edges(key)
    if edges:
        segs = [[(x0, y0), (x1, y1)] for (x0, y0), (x1, y1) in edges]
        lc   = LineCollection(segs, colors="#999999", linewidths=lw,
                              zorder=1, alpha=0.85)
        ax.add_collection(lc)

    valid = mae[mask & np.isfinite(mae)]
    vmax  = float(np.percentile(valid, 90)) if len(valid) else 10.0

    node_s  = 8  if key == "heusden" else 18
    sensor_s = 55 if key == "heusden" else 80

    sc = ax.scatter(cx[mask], cy[mask], c=mae[mask], cmap=_ERR_CMAP,
                    s=node_s, alpha=0.90, linewidths=0,
                    vmin=0, vmax=vmax, zorder=2)
    ax.scatter(cx[~mask], cy[~mask], c=C_SENSOR_SPATIAL,
               s=sensor_s, marker="*", linewidths=0.4,
               edgecolors="white", zorder=4,
               label=f"Sensors ({(~mask).sum()})")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.80, pad=0.02, label="MAE (cm)")
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="sci", scilimits=(5, 6), axis="both")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    pad_x = (cx.max() - cx.min()) * 0.04
    pad_y = (cy.max() - cy.min()) * 0.04
    ax.set_xlim(cx.min() - pad_x, cx.max() + pad_x)
    ax.set_ylim(cy.min() - pad_y, cy.max() + pad_y)

    return sc, vmax
